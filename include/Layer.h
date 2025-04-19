#ifndef LAYER_H
#define LAYER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include "Tile.h"

class Layer {
public:
    enum class LayerType { Background, Obstacle };

    Layer();
    ~Layer();

    // 設定圖層中 tile 的大小
    void setTileSize(int size);

    // 設定圖層類型
    void setLayerType(LayerType type);
    LayerType getLayerType() const;

    // 新增一整列 tile
    void addRow(const std::vector<Tile> &row);

    // 設定特定位置的 tile
    void setTile(int row, int col, const Tile &tile);

    // 檢查指定位置是否可通行（如果 tile 存在且為碰撞 tile 則回傳 false）
    bool isPassable(const sf::Vector2f &position) const;

    // 繪製整個圖層
    void draw(sf::RenderWindow &window) const;

private:
    std::vector< std::vector<Tile> > tiles;
    int tileSize;
    LayerType layerType;
};

#endif // LAYER_H
