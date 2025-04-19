#include "MapManager.h"
#include "Tile.h"
#include <iostream>

MapManager::MapManager() {
    loadMap();
}

MapManager::~MapManager() {}

void MapManager::loadMap() {
    // 1. 設定 tile 大小
    const int tileSize = 32;

    // 2. 設定整張地圖的寬度與高度（單位：格數）
    const int mapWidth = 20; // columns
    const int mapHeight = 15; // rows

    // 3. 載入草地與石頭貼圖
    sf::Texture *grassTexture = new sf::Texture();
    if (!grassTexture->loadFromFile("assets/env_img/grass.png")) {
        std::cerr << "無法載入 grass.png" << std::endl;
    }

    sf::Texture *stoneTexture = new sf::Texture();
    if (!stoneTexture->loadFromFile("assets//env_img/stone.png")) {
        std::cerr << "無法載入 stone.png" << std::endl;
    }

    // 4. 建立「背景層」：全部鋪草地
    Layer backgroundLayer;
    backgroundLayer.setTileSize(tileSize);
    backgroundLayer.setLayerType(Layer::LayerType::Background);

    for (int row = 0; row < mapHeight; ++row) {
        std::vector<Tile> rowTiles;
        for (int col = 0; col < mapWidth; ++col) {
            Tile tile;
            tile.setTexture(grassTexture, tileSize);
            tile.setPosition(col * tileSize, row * tileSize);
            tile.setCollidable(false);  // 草地可通行
            rowTiles.push_back(tile);
        }
        backgroundLayer.addRow(rowTiles);
    }
    
    
    // 5. 建立「障礙物層」：主要放邊界牆 + 右下偏移的牆
    
    Layer obstacleLayer;
    obstacleLayer.setTileSize(tileSize);
    obstacleLayer.setLayerType(Layer::LayerType::Obstacle);
    
    // 先建立空白的網格
    for (int row = 0; row < mapHeight; ++row) {
        std::vector<Tile> rowTiles;
        for (int col = 0; col < mapWidth; ++col) {
            Tile emptyTile;
            emptyTile.setCollidable(false);  // 預設不可碰撞
            rowTiles.push_back(emptyTile);
        }
        obstacleLayer.addRow(rowTiles);
    }
    
    // 6. 地圖四周：放牆壁
    for (int col = 0; col < mapWidth; ++col) {
        
        // 上邊 (row=0)
        Tile topWall;
        topWall.setTexture(stoneTexture, tileSize);
        topWall.setPosition(col * tileSize, 0);
        topWall.setCollidable(true);
        obstacleLayer.setTile(0, col, topWall);
        

        
        // 下邊 (row=mapHeight-1)
        Tile bottomWall;
        bottomWall.setTexture(stoneTexture, tileSize);
        bottomWall.setPosition(col * tileSize, (mapHeight - 1) * tileSize);
        bottomWall.setCollidable(true);
        obstacleLayer.setTile(mapHeight - 1, col, bottomWall);

        
    }

    for (int row = 1; row < mapHeight - 1; ++row) {
        // 左邊 (col=0)
        Tile leftWall;
        leftWall.setTexture(stoneTexture, tileSize);
        leftWall.setPosition(0, row * tileSize);
        leftWall.setCollidable(true);
        obstacleLayer.setTile(row, 0, leftWall);

        // 右邊 (col=mapWidth-1)
        Tile rightWall;
        rightWall.setTexture(stoneTexture, tileSize);
        rightWall.setPosition((mapWidth - 1) * tileSize, row * tileSize);
        rightWall.setCollidable(true);
        obstacleLayer.setTile(row, mapWidth - 1, rightWall);
    }
    
    // middle wall
    // 7. 在右下偏移位置放一段牆壁 (可自行調整形狀與範圍) 
    //   例如：在地圖的下半部，稍微靠右的地方，建一個 3x5 的牆區
    int wallStartRow = mapHeight / 2 - 3;   // 從地圖高度一半往下開始 // 4
    int wallStartCol = mapWidth / 2; // 從地圖寬度一半再往右偏一點 // 10
    int wallHeight = 8; // 高 3 格
    int wallWidth = 1;  // 寬 5 格

    for (int r = 0; r < wallHeight; ++r) {
        for (int c = 0; c < wallWidth; ++c) {
            int rowIdx = wallStartRow + r;
            int colIdx = wallStartCol + c;
            // 確保不超出地圖範圍
            if (rowIdx >= 0 && rowIdx < mapHeight &&
                colIdx >= 0 && colIdx < mapWidth) {
                Tile wallTile;
                wallTile.setTexture(stoneTexture, tileSize);
                wallTile.setPosition(colIdx * tileSize, rowIdx * tileSize);
                wallTile.setCollidable(true);
                obstacleLayer.setTile(rowIdx, colIdx, wallTile);
            }
        }
    }
    
    // top left wall
    int wallStartRow1 =  0;   // 從地圖高度一半往下開始
    int wallStartCol1 = mapWidth / 2 - 4; // 從地圖寬度一半再往右偏一點
    int wallHeight1 = 5; // 高 3 格
    int wallWidth1 = 1;  // 寬 5 格

    for (int r = 0; r < wallHeight1; ++r) {
        for (int c = 0; c < wallWidth1; ++c) {
            int rowIdx = wallStartRow1 + r;
            int colIdx = wallStartCol1 + c;
            // 確保不超出地圖範圍
            if (rowIdx >= 0 && rowIdx < mapHeight &&
                colIdx >= 0 && colIdx < mapWidth) {
                Tile wallTile;
                wallTile.setTexture(stoneTexture, tileSize);
                wallTile.setPosition(colIdx * tileSize, rowIdx * tileSize);
                wallTile.setCollidable(true);
                obstacleLayer.setTile(rowIdx, colIdx, wallTile);
            }
        }
    }

    // right down
    int wallStartRow2 = mapHeight - 5;   // 從地圖高度一半往下開始
    int wallStartCol2 = mapWidth / 2 + 4; // 從地圖寬度一半再往右偏一點
    int wallHeight2 = 5; // 高 3 格
    int wallWidth2 = 1;  // 寬 5 格

    for (int r = 0; r < wallHeight2; ++r) {
        for (int c = 0; c < wallWidth2; ++c) {
            int rowIdx = wallStartRow2 + r;
            int colIdx = wallStartCol2 + c;
            // 確保不超出地圖範圍
            if (rowIdx >= 0 && rowIdx < mapHeight &&
                colIdx >= 0 && colIdx < mapWidth) {
                Tile wallTile;
                wallTile.setTexture(stoneTexture, tileSize);
                wallTile.setPosition(colIdx * tileSize, rowIdx * tileSize);
                wallTile.setCollidable(true);
                obstacleLayer.setTile(rowIdx, colIdx, wallTile);
            }
        }
    }

    // right middle
    int wallStartRow3 = 5;   // 從地圖高度一半往下開始
    int wallStartCol3 = mapWidth / 2 + 5; // 從地圖寬度一半再往右偏一點
    int wallHeight3 = 1; // 高 3 格
    int wallWidth3 = 3;  // 寬 5 格

    for (int r = 0; r < wallHeight3; ++r) {
        for (int c = 0; c < wallWidth3; ++c) {
            int rowIdx = wallStartRow3 + r;
            int colIdx = wallStartCol3 + c;
            // 確保不超出地圖範圍
            if (rowIdx >= 0 && rowIdx < mapHeight &&
                colIdx >= 0 && colIdx < mapWidth) {
                Tile wallTile;
                wallTile.setTexture(stoneTexture, tileSize);
                wallTile.setPosition(colIdx * tileSize, rowIdx * tileSize);
                wallTile.setCollidable(true);
                obstacleLayer.setTile(rowIdx, colIdx, wallTile);
            }
        }
    }


    // left middle wall
    int wallStartRow4 = mapHeight - 5;   // 從地圖高度一半往下開始
    int wallStartCol4 = mapWidth / 2 - 5; // 從地圖寬度一半再往右偏一點
    int wallHeight4 = 1; // 高 3 格
    int wallWidth4 = 3;  // 寬 5 格

    for (int r = 0; r < wallHeight4; ++r) {
        for (int c = 0; c < wallWidth4; ++c) {
            int rowIdx = wallStartRow4 + r;
            int colIdx = wallStartCol4 + c;
            // 確保不超出地圖範圍
            if (rowIdx >= 0 && rowIdx < mapHeight &&
                colIdx >= 0 && colIdx < mapWidth) {
                Tile wallTile;
                wallTile.setTexture(stoneTexture, tileSize);
                wallTile.setPosition(colIdx * tileSize, rowIdx * tileSize);
                wallTile.setCollidable(true);
                obstacleLayer.setTile(rowIdx, colIdx, wallTile);
            }
        }
    }
    // 8. 將兩個圖層加入 layers
    layers.push_back(backgroundLayer);
    layers.push_back(obstacleLayer);
}

void MapManager::draw(sf::RenderWindow &window) {
    for (auto &layer : layers) {
        layer.draw(window);
    }
}

bool MapManager::isPassable(const sf::Vector2f &position) const {
    // 檢查障礙物層是否可通行
    for (const auto &layer : layers) {
        if (layer.getLayerType() == Layer::LayerType::Obstacle) {
            if (!layer.isPassable(position)) {
                return false;
            }
        }
    }
    return true;
}
