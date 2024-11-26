/**
 * This example demonstrates how to use ROS2 to send low-level motor commands of unitree go2 robot
 **/
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/motor_cmd.hpp"
#include "unitree_go/msg/bms_cmd.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"
#include "motor_crc.h"
using std::placeholders::_1;

// Create a low_level_cmd_sender class for low state receive
class low_level_cmd_sender : public rclcpp::Node
{
    public:
        low_level_cmd_sender() : Node("low_level_cmd_sender")
        {
            // the cmd_puber is set to subscribe "/lowcmd" topic
            cmd_puber = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);

            // The timer is set to 200Hz, and bind to low_level_cmd_sender::timer_callback function
            timer_ = this->create_wall_timer(std::chrono::milliseconds(int(dt * 1000)), std::bind(&low_level_cmd_sender::timer_callback, this));

            auto topic_low_name = "lowstate";
            auto topic_high_name = "sportmodestate";
            
            suber_low_state = this->create_subscription<unitree_go::msg::LowState>(
                topic_low_name, 10, std::bind(&low_level_cmd_sender::topic_low_callback, this, _1));

            suber_high_state = this->create_subscription<unitree_go::msg::SportModeState>(
                topic_high_name, 10, std::bind(&low_level_cmd_sender::topic_high_callback, this, _1));

            // Initialize lowcmd
            init_cmd();
        }
    private:
    void topic_high_callback(unitree_go::msg::SportModeState::SharedPtr data)
    {
        // Info motion states
        // Robot velocity (Odometry frame)

        RCLCPP_INFO(this->get_logger(), "Velocity -- vx: %f; vy: %f; vz: %f; yaw: %f",
                    data->velocity[0], data->velocity[1], data->velocity[2], data->yaw_speed);
    
        // Create the suber to receive motion states of robot
        rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr suber;
    };
    private:
        void topic_low_callback(unitree_go::msg::LowState::SharedPtr data)
        {
           
            imu_ = data->imu_state;
                               
            RCLCPP_INFO(this->get_logger(), "Euler angle -- rolsl: %f; pitch: %f; yaw: %f", imu_.rpy[0], imu_.rpy[1], imu_.rpy[2]);
            RCLCPP_INFO(this->get_logger(), "Quaternion -- qw: %f; qx: %f; qy: %f; qz: %f",
                        imu_.quaternion[0], imu_.quaternion[1], imu_.quaternion[2], imu_.quaternion[3]);
            RCLCPP_INFO(this->get_logger(), "Gyroscope -- wx: %f; wy: %f; wz: %f", imu_.gyroscope[0], imu_.gyroscope[1], imu_.gyroscope[2]);
            RCLCPP_INFO(this->get_logger(), "Accelerometer -- ax: %f; ay: %f; az: %f",
                        imu_.accelerometer[0], imu_.accelerometer[1], imu_.accelerometer[2]);
        
            for (int i = 0; i < 12; i++)
            {
                motor_[i] = data->motor_state[i];
                
                RCLCPP_INFO(this->get_logger(), "Motor state -- num: %d; q: %f; dq: %f; ddq: %f; tau: %f",
                                i, motor_[i].q, motor_[i].dq, motor_[i].ddq, motor_[i].tau_est);
            }
            
        
        }

    private:
        void timer_callback()
        {

            runing_time += dt;
            if (runing_time < 3.0)
            {
                // Stand up in first 3 second

                // Total time for standing up or standing down is about 1.2s
                phase = tanh(runing_time / 1.2);
                for (int i = 0; i < 12; i++)
                {
                    low_cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (1 - phase) * stand_down_joint_pos[i];
                    low_cmd.motor_cmd[i].dq = 0;
                    low_cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0;
                    low_cmd.motor_cmd[i].kd = 3.5;
                    low_cmd.motor_cmd[i].tau = 0;
                }
            }
            else
            {
                // Then stand down
                phase = tanh((runing_time - 3.0) / 1.2);
                for (int i = 0; i < 12; i++)
                {
                    low_cmd.motor_cmd[i].q = phase * stand_down_joint_pos[i] + (1 - phase) * stand_up_joint_pos[i];
                    low_cmd.motor_cmd[i].dq = 0;
                    low_cmd.motor_cmd[i].kp = 50;
                    low_cmd.motor_cmd[i].kd = 3.5;
                    low_cmd.motor_cmd[i].tau = 0;
                }
            }

            get_crc(low_cmd);            // Check motor cmd crc
            cmd_puber->publish(low_cmd); // Publish lowcmd message
        }

    void init_cmd()
    {

        for (int i = 0; i < 20; i++)
        {
            low_cmd.motor_cmd[i].mode = 0x01; // Set toque mode, 0x00 is passive mode
            low_cmd.motor_cmd[i].q = 
            PosStopF;
            low_cmd.motor_cmd[i].kp = 0;
            low_cmd.motor_cmd[i].dq = VelStopF;
            low_cmd.motor_cmd[i].kd = 0;
            low_cmd.motor_cmd[i].tau = 0;
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;                             // ROS2 timer
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr cmd_puber; // ROS2 Publisher
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr suber_low_state;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr suber_high_state;

    unitree_go::msg::LowCmd low_cmd;
    unitree_go::msg::IMUState imu_;
    unitree_go::msg::MotorState motor_[12];

    double stand_up_joint_pos[12] = {0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
                                     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763};
    double stand_down_joint_pos[12] = {0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
                                       1.22187, -2.44375, -0.0473455, 1.22187, -2.44375};
    double dt = 0.002;
    double runing_time = 0.0;
    double phase = 0.0;

};

int main(int argc, char **argv)
{   
    std::cout << "Press enter to start";
    std::cin.get();
    
    rclcpp::init(argc, argv);                             // Initialize rclcpp
    rclcpp::TimerBase::SharedPtr timer_;                  // Create a timer callback object to send cmd in time intervals
    auto node = std::make_shared<low_level_cmd_sender>(); // Create a ROS2 node and make share with low_level_cmd_sender class
    rclcpp::spin(node);                                   // Run ROS2 node
    rclcpp::shutdown();                                   // Exit
    return 0;
}
