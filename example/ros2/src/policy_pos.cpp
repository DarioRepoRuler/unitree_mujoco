#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"
#include "motor_crc.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <filesystem>

#include <cmath>

#define INFO_IMU 1          // Set 1 to info IMU states
#define INFO_MOTOR 1        // Set 1 to info motor states
#define INFO_FOOT_FORCE 0   // Set 1 to info foot force states
#define INFO_BATTERY 1      // Set 1 to info battery states
#define INFO_VELOCITY 1     // Set 1 to info base velocity states

#define HIGH_FREQ 1 // Set 1 to subscribe to low states with high frequencies (500Hz)

using std::placeholders::_1;

// Global variables
int policy_id = 0;                // Initialize policy_id
float policy_commands[3] = {0.0, 0.0, 0.0}; // Initialize policy_values

// ----------------------- Low Level Node -----------------------

class LowLevelNode : public rclcpp::Node
{
public:
    LowLevelNode(std::shared_ptr<torch::jit::script::Module> model) 
            : Node("low_level_node"),  
            dt_fast(0.002),
            dt_slow(0.02),
            runing_time_slow(0.0),
            runing_time_fast(0.0),
            phase(0.0),
            control_type("P"), 
            model(model)    
        {
        // Set up subscriber
        auto topic_low_name = "/lowstate";
        if (HIGH_FREQ)
        {
            topic_low_name = "lowstate";
        }
        auto topic_high_name = "lf/sport_mode_state";
        if (HIGH_FREQ)
        {
        topic_high_name = "sportmodestate";
        }
        suber_low_state = this->create_subscription<unitree_go::msg::LowState>(
            topic_low_name, 10, std::bind(&LowLevelNode::topic_low_callback, this, _1));

        suber_high_state = this->create_subscription<unitree_go::msg::SportModeState>(
            topic_high_name, 10, std::bind(&LowLevelNode::topic_high_callback, this, _1));

        // Set up publisher
        cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);

        // Set up timer
        // timer_ = this->create_wall_timer(
        //     std::chrono::milliseconds(int(dt * 1000)),
        //     std::bind(&LowLevelNode::timer_callback, this));
        
        timer_slow_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_slow),
            std::bind(&LowLevelNode::slow_timer_callback, this));
        
        timer_fast_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_fast),
            std::bind(&LowLevelNode::fast_timer_callback, this));

        if (control_type=="P") offset = 0;
        else if (control_type=="VIC_1") offset = 3;
        else if (control_type=="VIC_2") offset = 4;
        else if (control_type=="VIC_3") offset = 12;


        // Initialize lowcmd
        init_cmd();
        init_obs();
    }

private:
    void topic_low_callback(unitree_go::msg::LowState::SharedPtr data)
    {
        if (INFO_IMU)
        {   
            imu_ = data->imu_state;
            for (int i = 0; i < 4; i++)
            {
                quaternion[i] = imu_.quaternion[i];
            }
            float proj_gravity[3];
            this->get_gravity_vector(quaternion, proj_gravity);
            //RCLCPP_INFO(this->get_logger(), "Projected gravity vector: %f; %f; %f", proj_gravity[0], proj_gravity[1], proj_gravity[2]);
            float ang_vel[3] = {imu_.gyroscope[0], imu_.gyroscope[1], imu_.gyroscope[2]};

            //RCLCPP_INFO(this->get_logger(), "Accelerometer -- ang_vel: %f; %f; %f", ang_vel[0]*0.25, ang_vel[1]*0.25, ang_vel[2]*0.25);
            for (int i =0; i < 3; i++)
            {
                observations[i+6] = proj_gravity[i];
                observations[i+3] = ang_vel[i] * 0.25;
            }
             

            //RCLCPP_INFO(this->get_logger(), "Euler angle -- rolsl: %f; pitch: %f; yaw: %f", imu_.rpy[0], imu_.rpy[1], imu_.rpy[2]);
            // RCLCPP_INFO(this->get_logger(), "Quaternion -- qw: %f; qx: %f; qy: %f; qz: %f",
            //             imu_.quaternion[0], imu_.quaternion[1], imu_.quaternion[2], imu_.quaternion[3]);
            // RCLCPP_INFO(this->get_logger(), "Gyroscope -- wx: %f; wy: %f; wz: %f", imu_.gyroscope[0], imu_.gyroscope[1], imu_.gyroscope[2]);
            // RCLCPP_INFO(this->get_logger(), "Accelerometer -- ax: %f; ay: %f; az: %f",
            //             imu_.accelerometer[0], imu_.accelerometer[1], imu_.accelerometer[2]);
        }

        if (INFO_MOTOR)
        {   

            for (int i = 0; i < 12; i++)
            {
                motor_[i] = data->motor_state[i];
                observations[i+9] = motor_[i].q-default_motor_pos[i];
                observations[i+21] = motor_[i].dq *0.1;
                // RCLCPP_INFO(this->get_logger(), "Motor state -- num: %d; q: %f; dq: %f; ddq: %f; tau: %f",
                //             i, motor_[i].q, motor_[i].dq, motor_[i].ddq, motor_[i].tau_est);
            }
        }

        if (INFO_FOOT_FORCE)
        {
            for (int i = 0; i < 4; i++)
            {
                foot_force_[i] = data->foot_force[i];
                foot_force_est_[i] = data->foot_force_est[i];
            }

            // RCLCPP_INFO(this->get_logger(), "Foot force -- foot0: %d; foot1: %d; foot2: %d; foot3: %d",
            //             foot_force_[0], foot_force_[1], foot_force_[2], foot_force_[3]);
            // RCLCPP_INFO(this->get_logger(), "Estimated foot force -- foot0: %d; foot1: %d; foot2: %d; foot3: %d",
            //             foot_force_est_[0], foot_force_est_[1], foot_force_est_[2], foot_force_est_[3]);
        }

        if (INFO_BATTERY)
        {
            battery_current_ = data->power_a;
            battery_voltage_ = data->power_v;
            // RCLCPP_INFO(this->get_logger(), "Battery state -- current: %f; voltage: %f", battery_current_, battery_voltage_);
        }

        // for (int i=0; i<3; i++)
        // {
        //     observations[i+45] = policy_commands[i]*2.0;
        // }
        // RCLCPP_INFO(this->get_logger(), "Policy commands: [%f, %f, %f]", policy_commands[0], policy_commands[1], policy_commands[2]);
        
    }

    void topic_high_callback(unitree_go::msg::SportModeState::SharedPtr data)
    {
        float glob_vel[3]  = {data->velocity[0], data->velocity[1], data->velocity[2]};
        float local_v[3];
        float quat_inv[4];
        quat_invert(quaternion, quat_inv);
        rotate_vector(quat_inv, glob_vel, local_v);

        for (int i = 0; i < 3; i++)
        {
            observations[i] = 0.0;//local_v[i]*2.0;
        }

        RCLCPP_INFO(this->get_logger(), "Base velocity -- vx: %f; vy: %f; vz: %f", local_v[0], local_v[1], local_v[2]);
        // RCLCPP_INFO(this->get_logger(), "Position -- x: %f; y: %f; z: %f; body height: %f",
        //         data->position[0], data->position[1], data->position[2], data->body_height);
        // RCLCPP_INFO(this->get_logger(), "Velocity -- vx: %f; vy: %f; vz: %f; yaw: %f",
        //         data->velocity[0], data->velocity[1], data->velocity[2], data->yaw_speed);
    }

    void slow_timer_callback()
    {   
        
        for (int i=0; i<3; i++)
        {
            observations[i+45+offset] = policy_commands[i]*2.0;
        }
        //RCLCPP_INFO(this->get_logger(), "Policy commands: [%f, %f, %f]", policy_commands[0], policy_commands[1], policy_commands[2]);

        runing_time_slow += dt_slow;
        //RCLCPP_INFO(this->get_logger(), "\nSlow timer callback\n");
        if(policy_id ==0)
        {
            // Preprocess input data
            preprocess_input();
            
            // Run the model
            run_model();

            // Postprocess and apply model output
            postprocess_output();

            for (int i = 0;i<12;i++){
                if ( (i == 0) || (i == 3) || (i == 6) || (i == 9) ){
                    target_dof_pos[i] = actions[i]*action_scale*hip_scale+ default_motor_pos[i];
                }
                else{
                    target_dof_pos[i] = actions[i]*action_scale + default_motor_pos[i];
                }
            }
            if (control_type == "VIC_1"){
                for (int i = 0;i<3;i++){ //  every additional stiffness
                    for (int j = 0;j<4;j++){ // every foot
                        
                        target_kp[i+j] = std_stiffness* (m + r * actions[i+12]);
                        target_kd[i+j] = 0.2 * sqrt(target_kp[i+j]);
                    }                  
                }
            }
            if (control_type == "VIC_2"){
                for (int i = 0;i<4;i++){ //  i is the foot
                    for (int j = 0;j<3;j++){ // j is the group hip/thigh/calf
                        target_kp[i*3+j] = std_stiffness* (m + r * actions[i+12]);
                        target_kd[i*3+j] = 0.2 * sqrt(target_kp[i+j]);
                        //std::cout<< "Index" << i+j << " | Stiffness: " << target_kp[i*3+j] << " | Damping: " << target_kd[i*3+j] << std::endl;
                    }                  
                }
                
            }
            if (control_type=="VIC_3"){
                for (int i = 0;i<12;i++){
                    target_kp[i] = std_stiffness* (m +r * actions[i+12]);
                    target_kd[i] = 0.2 * sqrt(target_kp[i]);
                }
            }
        }

        else
        {
            for (int i = 0;i<12+offset;i++){
                actions[i] =0.0;
            }
        }
        
        
    }

    void fast_timer_callback()
    {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        runing_time_fast += dt_fast;
        if (runing_time_fast < 3.0 && policy_id ==0)
        {
            phase = tanh(runing_time_fast / 1.2);
            for (int i = 0; i < 12; i++)
            {
                low_cmd_.motor_cmd[i].q = phase * default_motor_pos[i] + (1 - phase) * stand_down_joint_pos_[i];
                low_cmd_.motor_cmd[i].dq = 0;
                low_cmd_.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0;
                low_cmd_.motor_cmd[i].kd = 1.0;
                low_cmd_.motor_cmd[i].tau = 0;
            }
        }

        else if (policy_id == 1)
        {   
            //RCLCPP_INFO(this->get_logger(), "[L]aying Down");

            if (time_offset < 0.0) {
                time_offset = runing_time_fast;
            }
            phase = tanh((runing_time_fast - time_offset) / 1.2);
            for (int i = 0; i < 12; i++)
            {
                low_cmd_.motor_cmd[i].q = phase * stand_down_joint_pos_[i] + (1 - phase) * default_motor_pos[i];
                low_cmd_.motor_cmd[i].dq = 0;
                low_cmd_.motor_cmd[i].kp =30.0;
                low_cmd_.motor_cmd[i].kd = 3.6;
                low_cmd_.motor_cmd[i].tau = 0;
            }
            if (runing_time_fast > time_offset +2.0){
                time_offset = -1.0;
                runing_time_fast = 0.0;
                policy_id = -1;
            }
        }
        else if (runing_time_fast > 3.0 && policy_id == 0)
        {   
            if (control_type == "P"){
                for (int i = 0; i < 12; i++)
                {
                    low_cmd_.motor_cmd[i].q = target_dof_pos[i];
                    low_cmd_.motor_cmd[i].dq = 0;
                    low_cmd_.motor_cmd[i].kp = std_stiffness;
                    low_cmd_.motor_cmd[i].kd = std_damp;
                    low_cmd_.motor_cmd[i].tau = 0;
                }
            }
            
            else if (control_type=="VIC_1" || control_type=="VIC_2" || control_type=="VIC_3"){
                //std::cout << "Target Dof pos: | Target Stiff | Target Damp" << std::endl;
                for (int i = 0; i < 12; i++)
                {
                    //std::cout << target_dof_pos[i]<< "|" << target_kp[i] << "|" << target_kd[i] << std::endl;
                    low_cmd_.motor_cmd[i].q = target_dof_pos[i];
                    low_cmd_.motor_cmd[i].dq = 0;
                    low_cmd_.motor_cmd[i].kp = target_kp[i];
                    low_cmd_.motor_cmd[i].kd = target_kd[i];
                    low_cmd_.motor_cmd[i].tau = 0;
                }
            }
            
        }

        if(policy_id == -1)
        {
            //RCLCPP_INFO(this->get_logger(), "StandBy");
            for (int i = 0; i < 12; i++)
            {
                low_cmd_.motor_cmd[i].q = 0.0;
                low_cmd_.motor_cmd[i].dq = 0;
                low_cmd_.motor_cmd[i].kp = 0;
                low_cmd_.motor_cmd[i].kd =3.5;
                low_cmd_.motor_cmd[i].tau = 0;
            }
            runing_time_fast = 0.0;
        }
        

        
        get_crc(low_cmd_);            // Check motor cmd crc
        cmd_puber_->publish(low_cmd_); // Publish lowcmd message
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        // RCLCPP_INFO(this->get_logger(), "Global Policy ID: %d", policy_id);
        // RCLCPP_INFO(this->get_logger(), "Global Policy Values: [%f, %f, %f]", policy_commands[0], policy_commands[1], policy_commands[2]);
        //RCLCPP_INFO(this->get_logger(), "Runing time fast: %f", runing_time_fast);
        //RCLCPP_INFO(this->get_logger(), "Time to publish command: %ld microseconds", duration);
        // Print observations
        print_observations();
    }
    
    


    void preprocess_input()
    {
        // Convert observations to tensor
        //std::cout << "Offset: " << offset << std::endl;
        model_input = torch::from_blob(observations, {1, 48+offset}, torch::kFloat32);
        //std::cout << "Model input: " << model_input << std::endl;
        model_input = model_input.clone(); // Ensure tensor is not shared
    }

    void run_model()
    {
        // Run the model
        model_output = model->forward({model_input}).toTensor();
    }

    void postprocess_output()
    {
        // Convert tensor to array and use the output
        auto output_data = model_output.accessor<float, 2>();
        // Example: use the output data

        for (int i = 0; i < 12+offset; ++i)
            {
                actions[i] = output_data[0][i]; // Adjust as needed based on model output
                observations[i+33] = actions[i];
            }

    }

    private:
    void print_observations()
    {
        std::string observations_str = "Observations: ";

        // Loop through the global array and append each value to the string
        for (size_t i = 0; i < sizeof(observations) / sizeof(observations[0]); ++i) {
            observations_str += std::to_string(observations[i]);
            if (i < sizeof(observations) / sizeof(observations[0]) - 1) {
                observations_str += ", "; // Add a comma between values
            }
        }

        // Print the concatenated string using RCLCPP_INFO
        //RCLCPP_INFO(this->get_logger(), "%s", observations_str.c_str());
    }

    private:
    // Function to compute the inverse of a quaternion
    void quat_invert(const float q[4], float q_inv[4])
    {
        q_inv[0] = q[0];  // w
        q_inv[1] = -q[1]; // -x
        q_inv[2] = -q[2]; // -y
        q_inv[3] = -q[3]; // -z
    }

    // Function to compute the rotation matrix from a quaternion
    void quaternion_to_rotation_matrix(const float q[4], float R[3][3])
    {
        float q_w = q[0];
        float q_x = q[1];
        float q_y = q[2];
        float q_z = q[3];

        R[0][0] = 1 - 2 * (q_y * q_y + q_z * q_z);
        R[0][1] = 2 * (q_x * q_y - q_w * q_z);
        R[0][2] = 2 * (q_x * q_z + q_w * q_y);

        R[1][0] = 2 * (q_x * q_y + q_w * q_z);
        R[1][1] = 1 - 2 * (q_x * q_x + q_z * q_z);
        R[1][2] = 2 * (q_y * q_z - q_w * q_x);

        R[2][0] = 2 * (q_x * q_z - q_w * q_y);
        R[2][1] = 2 * (q_y * q_z + q_w * q_x);
        R[2][2] = 1 - 2 * (q_x * q_x + q_y * q_y);
    }

    // Function to rotate a vector using a rotation matrix
    void rotate_vector(const float q[4], const float g_world[3], float g_body[3])
    {
        float R[3][3];
        quaternion_to_rotation_matrix(q, R);

        for (int i = 0; i < 3; ++i)
        {
            g_body[i] = 0.0;
            for (int j = 0; j < 3; ++j)
            {
                g_body[i] += R[i][j] * g_world[j];
            }
        }
    }

    // Function to get the gravity vector
    void get_gravity_vector(const float quaternion[4], float proj_grav[3])
    {
        float g[3] = {0.0, 0.0, -1.0}; // Gravity vector
        float quaternion_inv[4];

        quat_invert(quaternion, quaternion_inv);
        rotate_vector(quaternion_inv, g, proj_grav);
    }

    private:
    void init_cmd()
    {
        for (int i = 0; i < 20; i++)
        {
            low_cmd_.motor_cmd[i].mode = 0x01; // Set torque mode, 0x00 is passive mode
            low_cmd_.motor_cmd[i].q = PosStopF;
            low_cmd_.motor_cmd[i].kp = 0;
            low_cmd_.motor_cmd[i].dq = VelStopF;
            low_cmd_.motor_cmd[i].kd = 0;
            low_cmd_.motor_cmd[i].tau = 0;
        }
    }
    void init_obs()
    {
        for (int i = 0; i < 48; i++)
        {
            observations[i] = 0;
        }
    }

    // ROS2 publisher and subscriber
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr suber_low_state;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr suber_high_state;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr cmd_puber_;
    //rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr timer_slow_;
    rclcpp::TimerBase::SharedPtr timer_fast_;

    // Data members for subscriber
    unitree_go::msg::IMUState imu_;
    unitree_go::msg::MotorState motor_[12];
    int16_t foot_force_[4];
    int16_t foot_force_est_[4];
    float battery_voltage_;
    float battery_current_;
    float observations[52];
    float quaternion[4];
    float target_dof_pos[12];
    float target_kp[12];
    float target_kd[12];
    //float rotation_matrix[3][3]; // just for rotation 
    float gravity_world[3] = {0.0, 0.0, -1.0};
    float actions[24];
    float std_stiffness = 50.0;
    float std_damp = 1.0;
    float m = (1.5 + 0.5)/2;
    float r = (1.5 - 0.5)/2;
    
    float action_scale = 0.5;
    float hip_scale = 0.6;

    // Data members for publisher
    unitree_go::msg::LowCmd low_cmd_;
    double stand_up_joint_pos_[12] = {-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8};
    double stand_down_joint_pos_[12] = {0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
                                        0.15, 1.22187, -2.44375, -0.15, 1.22187, -2.44375};
    // float default_motor_pos[12] = {-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8  };
    float default_motor_pos[12] = {-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5};
    double dt_fast;
    double dt_slow;
    double runing_time_slow; 
    double runing_time_fast;
    double time_offset = -1.0;
    double phase;
    int offset;
    std::string control_type;

    // PyTorch model
    std::shared_ptr<torch::jit::script::Module> model;
    torch::Tensor model_input;
    torch::Tensor model_output;

};



// ----------------------- For Keyboard Control -----------------------
int getch() {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();

    if (ch == 27) {  // ESC key
        if (getchar() == '[') {
            switch(getchar()) {
                case 'A': ch = 256 + 'A'; break; // Up arrow
                case 'B': ch = 256 + 'B'; break; // Down arrow
                case 'C': ch = 256 + 'C'; break; // Right arrow
                case 'D': ch = 256 + 'D'; break; // Left arrow
                // You can add more cases here for other special keys
            }
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}


int kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

std::atomic<bool> running(true);

void listenForKeyPress() {
    while (running) {
        if (kbhit()) {
            int ch = getch();
            if (ch == 'q' || ch == 'Q') {
                std::cout << "Quit command received. Exiting...\n";
                running = false;
                rclcpp::shutdown();  // Signal to shutdown ROS2
                break;
            }
            
            else if (ch == 's' || ch == 'S') {
                policy_id = 0;
                policy_commands[0] = 0.0;
                policy_commands[1] = 0.0;
                policy_commands[2] = 0.0;
            }
            else if (ch == 'l' || ch == 'L') {
                policy_id = 1;
                policy_commands[0] = 0.0;
                policy_commands[1] = 0.0;
                policy_commands[2] = 0.0;
            }
            else if (ch=='n' || ch == 'N') {
                policy_commands[0] = 0.0;
                policy_commands[1] = 0.0;
                policy_commands[2] = 0.0;
            }
            // Handle special keys
            else if (ch >= 256) {
                switch (ch) {
                    case 256 + 'A': policy_commands[0] =std::min(policy_commands[0]+0.025f, 1.2f); break;
                    case 256 + 'B': policy_commands[0] =std::max(policy_commands[0]-0.025f, -1.2f); break;
                    case 256 + 'C': policy_commands[1] =std::max(policy_commands[1]-0.05f, -1.2f); break;
                    case 256 + 'D': policy_commands[1] =std::min(policy_commands[1]+0.05f, 1.2f); break;
                    // Add more cases for other special keys if needed
                }
            } else {
                //std::cout << "Key pressed: " << static_cast<char>(ch) << "\n";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Small sleep to prevent busy waiting
    }
}

// ----------------------- Main Function -----------------------

int main(int argc, char **argv)
{
    torch::jit::script::Module model;
    try {
        // Get the path to the executable
        std::filesystem::path exePath = std::filesystem::canonical(argv[0]);
        std::filesystem::path modelPath = exePath.parent_path() / "p50_model.pt";
        std::cout << "Model path: " << modelPath.string() << std::endl;
        // Load the model
        model = torch::jit::load(modelPath.string());
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    std::cout << "Model loaded successfully\n";

    std::cout << "Press enter to start";
    std::cin.get();

    rclcpp::init(argc, argv);
    auto node = std::make_shared<LowLevelNode>(std::make_shared<torch::jit::script::Module>(model));

    // Start the keyboard listener thread
    std::thread keyListener(listenForKeyPress);

    rclcpp::spin(node);
    rclcpp::shutdown();

    // Wait for the keyboard listener thread to finish before exiting
    running = false;  // Stop the listener if it's still running
    keyListener.join();

    return 0;
}
