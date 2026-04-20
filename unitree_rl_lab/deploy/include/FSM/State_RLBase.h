// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        env->reset();

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            const auto start = clock::now();
            auto sleepTill = start + dt;

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::filesystem::path parser_policy_dir(std::filesystem::path policy_dir)
    {
        // Load Policy
        if (policy_dir.is_relative()) {
            policy_dir = param::proj_dir / policy_dir;
        }

        // If there is no `exported` folder in this folder,
        // then sort all the folders under this folder and take the last folder
        if (!std::filesystem::exists(policy_dir / "exported")) {
            auto dirs = std::filesystem::directory_iterator(policy_dir);
            std::vector<std::filesystem::path> dir_list;
            for (const auto& entry : dirs) {
                if (entry.is_directory()) {
                    dir_list.push_back(entry.path());
                }
            }
            if (!dir_list.empty()) {
                std::sort(dir_list.begin(), dir_list.end());
                // Check if there is an `exported` folder starting from the last folder
                for (auto it = dir_list.rbegin(); it != dir_list.rend(); ++it) {
                    if (std::filesystem::exists(*it / "exported")) {
                        policy_dir = *it;
                        break;
                    }
                }
            }
        }
        spdlog::info("Policy directory: {}", policy_dir.string());
        return policy_dir;
    }


    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};