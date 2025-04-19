#ifndef DECISIONMAKER_H
#define DECISIONMAKER_H

#include "Character.h"
#include "MapManager.h"

class DecisionMaker {
public:
    DecisionMaker();
    ~DecisionMaker();

    // 根據目前環境與角色狀態做出決策
    void makeDecision(Character &character, const MapManager &mapManager);
};

#endif // DECISIONMAKER_H
