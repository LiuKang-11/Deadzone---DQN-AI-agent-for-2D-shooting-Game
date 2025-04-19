#include "Layer.h"

Layer::Layer() : tileSize(64), layerType(LayerType::Background) {}

Layer::~Layer() {}

void Layer::setTileSize(int size) {
    tileSize = size;
}

void Layer::setLayerType(LayerType type) {
    layerType = type;
}

Layer::LayerType Layer::getLayerType() const {
    return layerType;
}

void Layer::addRow(const std::vector<Tile> &row) {
    tiles.push_back(row);
}

void Layer::setTile(int row, int col, const Tile &tile) {
    if (row < tiles.size() && col < tiles[row].size())
        tiles[row][col] = tile;
}

bool Layer::isPassable(const sf::Vector2f &position) const {
    // 根據位置計算 tile 的索引
    int col = static_cast<int>(position.x) / tileSize;
    int row = static_cast<int>(position.y) / tileSize;
    if (row < 0 || row >= tiles.size() || col < 0 || col >= tiles[row].size())
        return false;
    // 若該 tile 為可碰撞，則不可通行
    // 此處簡單檢查左上角點碰撞，後續可擴展
    return !tiles[row][col].checkCollision(position);
}

void Layer::draw(sf::RenderWindow &window) const {
    for (const auto &row : tiles) {
        for (const auto &tile : row) {
            tile.draw(window);
        }
    }
}
