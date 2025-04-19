#ifndef CHARACTER_H
#define CHARACTER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include "MapManager.h"

class Character {
public:
    static constexpr int ACTION_COUNT = 13;
    static constexpr float TURN_ANGLE = 5.f;

    Character();
    ~Character();

    // human control
    void update(const MapManager& mapManager, std::vector<Character>& characters);
    void draw(sf::RenderWindow& window);
    void rotate(float angleDelta);
    void setPosition(const sf::Vector2f& position);
    void setColor(const sf::Color color);

    // health
    int getHealth() const { return health; }
    void setHealth(int h) { health = h; }

    // expose for collision tests
    sf::FloatRect getGlobalBounds() const { return sprite.getGlobalBounds(); }

    // AI interfaces
    std::vector<float> serializeState(
        const MapManager& mapManager,
        const std::vector<Character>& characters
    ) const;
    void performAction(
        int action,
        const MapManager& mapManager,
        std::vector<Character>& characters
    );

private:
    void move(const sf::Vector2f& offset, const MapManager& mapManager);
    void shoot(const MapManager& mapManager, std::vector<Character>& characters);
    void switchWeapon();
    void drawHealthBar(sf::RenderWindow& window);

    sf::Texture rifleTexture, shotgunTexture;
    sf::Sprite sprite;
    sf::Vector2f rifleScale, shotgunScale;
    bool usingRifle = true;
    bool prevRightPressed = false;
    bool shooting = false;
    int currFireRate = 0, shootTimer = 0;
    int health = 10;

    sf::VertexArray bulletLine;
    sf::VertexArray shotgunLine1;
    sf::VertexArray shotgunLine2;
};

#endif // CHARACTER_H
