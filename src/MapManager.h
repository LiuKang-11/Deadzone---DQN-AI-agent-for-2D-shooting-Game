#ifndef MAPMANAGER_H
#define MAPMANAGER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include "Layer.h"

// MapManager 負責載入地圖資料與管理各圖層
class MapManager {
public:
    MapManager();
    ~MapManager();

    // 載入靜態地圖配置
    void loadMap();

    // 在視窗上渲染整個地圖
    void draw(sf::RenderWindow &window);

    // 檢查位置是否可通行（例如草地才可通行）
    bool isPassable(const sf::Vector2f &position) const;
    
    // ← Add these:
    int getWidth()  const { return mapWidth; }
    int getHeight() const { return mapHeight; }

private:
    // 地圖由多個圖層構成：例如背景層、障礙物層
    std::vector<Layer> layers;
    int mapWidth  = 20;  // number of columns
    int mapHeight = 15;  // number of rows
};

#endif // MAPMANAGER_H
