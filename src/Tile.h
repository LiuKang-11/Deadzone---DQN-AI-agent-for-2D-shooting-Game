#ifndef TILE_H
#define TILE_H

#include <SFML/Graphics.hpp>

class Tile {
public:
    Tile();
    ~Tile();

    // 設定圖塊所使用的 texture
    void setTexture(sf::Texture *texture, int tileSize);

    // 設定 tile 在畫面上的位置
    void setPosition(float x, float y);

    // 設定是否有碰撞性
    void setCollidable(bool collidable);

    // 繪製 tile
    void draw(sf::RenderWindow &window) const;

    // 檢查指定點是否與 tile 碰撞
    bool checkCollision(const sf::Vector2f &point) const;

private:
    sf::Sprite sprite;
    bool isCollidable;
};

#endif // TILE_H
