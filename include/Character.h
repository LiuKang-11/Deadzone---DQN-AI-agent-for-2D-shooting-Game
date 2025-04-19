// Character.h
#ifndef CHARACTER_H
#define CHARACTER_H

#include <SFML/Graphics.hpp>

class Character {
public:
    Character();
    ~Character();

    void setPosition(const sf::Vector2f &position);
    void update();
    void draw(sf::RenderWindow &window);

private:
    sf::Texture rifleTexture;
    sf::Texture shotgunTexture;
    sf::Sprite sprite;
    bool usingRifle;

    // 新增這兩個成員，用來記錄縮放比例
    sf::Vector2f rifleScale;
    sf::Vector2f shotgunScale;
};

#endif // CHARACTER_H
