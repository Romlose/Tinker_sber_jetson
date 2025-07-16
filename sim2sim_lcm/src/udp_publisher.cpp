#include "udp_publish.h"
#include <iostream>
#include <fstream>
#include <valarray>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <sys/shm.h>
#include <arpa/inet.h>
#include <time.h>
#include <errno.h>
using namespace std;
using namespace torch::indexing;

RL_Tinymal_UDP tinymal_rl;
struct _msg_request msg_request;
struct _msg_response msg_response;

float limit(float input, float min, float max) {
    if (input > max) return max;
    if (input < min) return min;
    return input;
}

void RL_Tinymal_UDP::handleMessage(_msg_request request) {
    std::cout << "Handling message with trigger: " << request.trigger << std::endl;
    if (request.trigger == 1) {
        request.trigger = 0;
        std::vector<float> obs;
        obs.push_back(request.omega[0] * omega_scale);
        obs.push_back(request.omega[1] * omega_scale);
        obs.push_back(request.omega[2] * omega_scale);
        obs.push_back(request.eu_ang[0] * eu_ang_scale);
        obs.push_back(request.eu_ang[1] * eu_ang_scale);
        obs.push_back(request.eu_ang[2] * eu_ang_scale);

        float max = 1.0, min = -1.0;
        cmd_x = cmd_x * (1 - smooth) + (std::fabs(request.command[0]) < dead_zone ? 0.0 : request.command[0]) * smooth;
        cmd_y = cmd_y * (1 - smooth) + (std::fabs(request.command[1]) < dead_zone ? 0.0 : request.command[1]) * smooth;
        cmd_rate = cmd_rate * (1 - smooth) + (std::fabs(request.command[2]) < dead_zone ? 0.0 : request.command[2]) * smooth;

        obs.push_back(cmd_x * lin_vel);
        obs.push_back(cmd_y * lin_vel);
        obs.push_back(cmd_rate * ang_vel);

        for (int i = 0; i < 12; ++i) {
            float pos = (request.q[i] - init_pos[i]) * pos_scale;
            obs.push_back(pos);
        }
        for (int i = 0; i < 12; ++i) {
            float vel = request.dq[i] * vel_scale;
            obs.push_back(vel);
        }
        for (int i = 0; i < 12; ++i) {
            obs.push_back(action_temp[i]);
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(), {1, 45}, options).to(device);
        auto obs_buf_batch = obs_buf.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs_tensor.to(torch::kHalf));
        inputs.push_back(obs_buf_batch.to(torch::kHalf));

        torch::Tensor action_tensor = model.forward(inputs).toTensor();
        action_buf = torch::cat({action_buf.index({Slice(1, None), Slice()}), action_tensor}, 0);

        torch::Tensor action_blend_tensor = 0.8 * action_tensor + 0.2 * last_action;
        last_action = action_tensor.clone();
        this->obs_buf = torch::cat({this->obs_buf.index({Slice(1, None), Slice()}), obs_tensor}, 0);

        torch::Tensor action_raw = action_blend_tensor.squeeze(0);
        action_raw = action_raw.to(torch::kFloat32).to(torch::kCPU);
        auto action_getter = action_raw.accessor<float, 1>();

        for (int j = 0; j < 12; j++) {
            action[j] = limit(action_getter[j], -5, 5);
            action_temp[j] = limit(action_getter[j], -5, 5);
        }
        action_refresh = 1;
    }
}

int RL_Tinymal_UDP::load_policy() {
    std::cout << model_path << std::endl;
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device = torch::kCPU;
    if (torch::cuda::is_available() && 1) {
        device = torch::kCUDA;
    }
    std::cout << "device:" << device << endl;
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    model.to(torch::kHalf);
    std::cout << "load model to device!" << std::endl;
    model.eval();
}

int RL_Tinymal_UDP::init_policy() {
    std::cout << "RL model thread start" << endl;
    cout << "cuda_is_available:" << torch::cuda::is_available() << endl;
    cout << "cudnn_is_available:" << torch::cuda::cudnn_is_available() << endl;

    model_path = "/Tinker_sber_jetson/model_jitt.pt";
    load_policy();

    action_buf = torch::zeros({history_length, 12}, device);
    obs_buf = torch::zeros({history_length, 45}, device);
    last_action = torch::zeros({1, 12}, device);

    action_buf.to(torch::kHalf);
    obs_buf.to(torch::kHalf);
    last_action.to(torch::kHalf);

    for (int j = 0; j < 12; j++) {
        action_temp.push_back(0.0);
        action.push_back(init_pos[j]);
        prev_action.push_back(init_pos[j]);
    }

    for (int i = 0; i < history_length; i++) {
        std::vector<float> obs;
        obs.insert(obs.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0});
        for (int i = 0; i < 12; ++i) obs.push_back(0);
        for (int i = 0; i < 12; ++i) obs.push_back(0);
        for (int i = 0; i < 12; ++i) obs.push_back(0);
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(), {1, 45}, options).to(device);
    }
}

int main(int argc, char** argv) {
    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        exit(1);
    }

    struct sockaddr_in addr_serv;
    int len;
    memset(&addr_serv, 0, sizeof(addr_serv));
    addr_serv.sin_family = AF_INET;
#if 0
    string UDP_IP = "127.0.0.1";
    int SERV_PORT = 8888;
#else
    string UDP_IP = "192.168.1.11";
    int SERV_PORT = 10000;
#endif
    addr_serv.sin_addr.s_addr = inet_addr(UDP_IP.c_str());
    addr_serv.sin_port = htons(SERV_PORT);
    len = sizeof(addr_serv);

    // Установка тайм-аута для сокета
    struct timeval timeout;
    timeout.tv_sec = 1;  // 1 секунда
    timeout.tv_usec = 0;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        perror("setsockopt failed");
        exit(1);
    }

    int recv_num = 0, send_num = 0;
    char send_buf[500] = {0}, recv_buf[500] = {0};

    tinymal_rl.init_policy();
    for (int i = 0; i < 12; i++)
        msg_response.q_exp[i] = tinymal_rl.action[i];

    printf("Thread UDP RL-Tinker\n");
    std::cout << "UDP IP is: " << UDP_IP << "\n";
    std::cout << "UDP address is: " << SERV_PORT << "\n";

    int loop_count = 0;
    std::cout << loop_count << "\n";

    std::ofstream log_file("UDP_logs.txt", std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error opening log file! Continuing without logging." << std::endl;
    }

    while (1) {
        loop_count++;
        std::cout << "Loop iteration: " << loop_count << std::endl;
        if (log_file.is_open()) {
            log_file << "Loop iteration: " << loop_count << std::endl;
        }

        if (tinymal_rl.action_refresh) {
            tinymal_rl.action_refresh = 0;
            for (int i = 0; i < 12; i++)
                msg_response.q_exp[i] = tinymal_rl.action[i];
                
            std::cout.precision(2);
            if (log_file.is_open()) log_file.precision(2);

    #if 1
            cout << endl << "act send:";
            if (log_file.is_open()) log_file << endl << "act send:";
            for (int i = 0; i < 12; i++) {
                cout << msg_response.q_exp[i] << " ";
                if (log_file.is_open()) log_file << msg_response.q_exp[i] << " ";
            }
            cout << endl;
            if (log_file.is_open()) log_file << endl;
    #endif
        }

        if (send_num < 0) {
            std::string error_msg = "Robot sendto error: " + std::string(strerror(errno));
            std::cerr << error_msg << std::endl;
            if (log_file.is_open()) log_file << error_msg << std::endl;
            exit(1);
        } else {
            std::cout << "Sent " << send_num << " bytes to " << UDP_IP << ":" << SERV_PORT << std::endl;
            if (log_file.is_open()) {
                log_file << "Sent " << send_num << " bytes to " << UDP_IP << ":" << SERV_PORT << std::endl;
            }
        }

        if (recv_num < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::cout << "recvfrom timed out after 1 second" << std::endl;
                if (log_file.is_open()) log_file << "recvfrom timed out after 1 second" << std::endl;
            } else {
                std::string error_msg = "recvfrom error: " + std::string(strerror(errno));
                std::cerr << error_msg << std::endl;
                if (log_file.is_open()) log_file << error_msg << std::endl;
                exit(1);
            }
        } else {
            std::cout << "Received " << recv_num << " bytes" << std::endl;
            if (log_file.is_open()) log_file << "Received " << recv_num << " bytes" << std::endl;
        }
        usleep(5 * 1000);
    }
    return 0;
}