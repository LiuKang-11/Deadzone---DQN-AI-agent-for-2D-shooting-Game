// main.cpp
#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/Network.hpp>
#include "MapManager.h"
#include "Character.h"
#include <vector>

// 发送状态、接收 AI 动作
int sendStateAndReceiveAction(
    const std::vector<float>& state,
    sf::TcpSocket& socket
) {
    std::string msg;
    for (auto v: state) msg += std::to_string(v)+",";
    if (!msg.empty()) msg.pop_back();
    std::cout << "[DEBUG] Sending state: " << msg << std::endl;
    sf::Packet pkt;
    pkt << msg;
    if (socket.send(pkt)!=sf::Socket::Done) return -1;

    sf::Packet reply;
    if (socket.receive(reply)!=sf::Socket::Done) return -1;

    std::string actStr;
    reply >> actStr;
    return std::stoi(actStr);
}

int main() {
    // —— SFML 窗口 & 文字初始化 ——
    sf::RenderWindow window(sf::VideoMode(640, 480), "2D Shooter");
    window.setKeyRepeatEnabled(false);

    sf::Font font;
    if (!font.loadFromFile("assets/fonts/Danger Zone Warning.ttf"))
        std::cerr<<"Cannot find font file\n";
    sf::Text finishText("CONGRATULATIONS!", font, 50);
    finishText.setPosition(85, 200);

    // —— 地图 & 角色初始化 ——
    MapManager mapManager;
    Character player;
    player.setColor(sf::Color::White);

    Character opponent;
    opponent.setPosition({300.f,300.f});
    opponent.setColor(sf::Color::Red);

    std::vector<Character> characters;
    characters.push_back(player);
    characters.push_back(opponent);

    // —— AI Socket 连接 —— 
    sf::TcpSocket socket;
    if (socket.connect("127.0.0.1", 5001) == sf::Socket::Done) {
        socket.setBlocking(false);
        std::cout<<"AI connected\n";
    } else {
        std::cerr<<"AI connect failed\n";
    }

    // —— 视图设置 —— 
    const int tileSize = 32, mapW = 20, mapH = 15;
    sf::View view({0,0, (float)window.getSize().x, (float)window.getSize().y});
    view.setCenter(mapW*tileSize/2.f, mapH*tileSize/2.f);
    window.setView(view);

    // —— 主循环 —— 
    while (window.isOpen()) {
        // 1) 处理系统事件
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) window.close();
            // 这里不再处理 Tab/Space 旋转，交给 AI
        }

        // 2) **AI 控制** —— 先序列化状态，发给 Python Server，收动作
        for (int i = 0; i < 2; ++i) {
            auto state = characters[i].serializeState(mapManager, characters);
            std::string prefix = (i == 0 ? "P0:" : "P1:");
            std::string msg = prefix;
            for (auto v: state) msg += std::to_string(v)+",";
            if (!msg.empty()) msg.pop_back();

            sf::Packet pkt;
            pkt << msg;
            if (socket.send(pkt) != sf::Socket::Done) continue;

            sf::Packet reply;
            if (socket.receive(reply) != sf::Socket::Done) continue;

            std::string actStr;
            reply >> actStr;
            int action = std::stoi(actStr);
            std::cout << "[DEBUG] P" << i << " action = " << action << std::endl;
            if (action >= 0 && action < Character::ACTION_COUNT)
                characters[i].performAction(action, mapManager, characters);
}



        // 4) 绘制
        window.clear();
        if (characters.size() > 1) {
            mapManager.draw(window);
            // 注意：performAction 里可能把对手打死，要在 draw 前剔除
            for (int i = 0; i < (int)characters.size(); ++i) {
                if (characters[i].getHealth() <= 0) {
                    characters.erase(characters.begin()+i);
                    --i;
                } else {
                    characters[i].draw(window);
                }
            }
        } else {
            window.draw(finishText);
        }
        window.display();
    }

    return 0;
}