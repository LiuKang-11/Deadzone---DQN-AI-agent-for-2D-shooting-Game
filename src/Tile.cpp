#include "Tile.h"

Tile::Tile() : isCollidable(false) {}

Tile::~Tile() {}

void Tile::setTexture(sf::Texture *texture, int tileSize) {
    sprite.setTexture(*texture);
    float scaleX = static_cast<float>(tileSize) / texture->getSize().x;
    float scaleY = static_cast<float>(tileSize) / texture->getSize().y;
    sprite.setScale(scaleX, scaleY);
}

void Tile::setPosition(float x, float y) {
    sprite.setPosition(x, y);
}

void Tile::setCollidable(bool collidable) {
    isCollidable = collidable;
}

void Tile::draw(sf::RenderWindow &window) const {
    window.draw(sprite);
}

bool Tile::checkCollision(const sf::Vector2f &point) const {
    return isCollidable && sprite.getGlobalBounds().contains(point);
}
