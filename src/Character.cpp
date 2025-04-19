// Character.cpp
#include "Character.h"
#include <cmath>
#include <iostream>
#include <SFML/Window/Keyboard.hpp>
#include <cassert>
#include <iostream>


// 常數定義
static const float DESIRED_HEIGHT    = 50.f;
static const float PI                = 3.141592653589793f;
static const float RIFLE_RANGE       = 300.f;
static const float SHOTGUN_RANGE     = 100.f;
static const int   RIFLE_FIRE_RATE   = 400;
static const int   SHOTGUN_FIRE_RATE = 800;
static const int   RIFLE_DAMAGE      = 1;
static const int   SHOTGUN_DAMAGE    = 3;

Character::Character()
: bulletLine(sf::Lines,2)
, shotgunLine1(sf::Lines,2)
, shotgunLine2(sf::Lines,2)
{
    // 載入貼圖
    if (!rifleTexture.loadFromFile("assets/character/survivor_rifle.png"))
        std::cerr<<"can't load survivor_rifle.png\n";
    if (!shotgunTexture.loadFromFile("assets/character/survivor_shotgun.png"))
        std::cerr<<"can't load survivor_shotgun.png\n";

    // 計算縮放比
    sf::Sprite tmp(rifleTexture);
    float scale = DESIRED_HEIGHT / tmp.getLocalBounds().height;
    rifleScale  = {scale, scale};
    tmp.setTexture(shotgunTexture);
    scale = DESIRED_HEIGHT / tmp.getLocalBounds().height;
    shotgunScale = {scale, scale};

    // 初始 sprite
    sprite.setTexture(rifleTexture);
    sprite.setScale(rifleScale);
    sprite.setPosition(100.f,100.f);
    sprite.setOrigin(100.f,100.f);
}

Character::~Character() {}

void Character::setPosition(const sf::Vector2f& pos) {
    sprite.setPosition(pos);
}

void Character::setColor(const sf::Color color) {
    sprite.setColor(color);
}

void Character::update(const MapManager& mapManager, std::vector<Character>& characters) {
    bool currR = sf::Mouse::isButtonPressed(sf::Mouse::Right);
    bool currL = sf::Mouse::isButtonPressed(sf::Mouse::Left);

    // 人類鍵盤移動
    sf::Vector2f mv{0,0};
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))    mv.y -= 0.1f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))  mv.y += 0.1f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  mv.x -= 0.1f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) mv.x += 0.1f;
    if (mv.x||mv.y) move(mv, mapManager);

    // 換槍
    if (currR && !prevRightPressed) switchWeapon();

    // 人類射擊
    if (currL) shoot(mapManager, characters);

    prevRightPressed = currR;
}

void Character::move(const sf::Vector2f& off, const MapManager& mapManager) {
    auto newPos = sprite.getPosition()+off;
    if (mapManager.isPassable(newPos)) {
        sprite.move(off);
        // 根據方向旋轉
        //if (off.x>0) sprite.setRotation(0);
        //else if (off.x<0) sprite.setRotation(180);
        //else if (off.y>0) sprite.setRotation(90);
        //else if (off.y<0) sprite.setRotation(270);
    }
}

void Character::shoot(const MapManager& mapManager, std::vector<Character>& characters) {
    // --- 判断这帧是否真的要发射 ---
    bool canFireThisFrame = 
        (usingRifle   && currFireRate % RIFLE_FIRE_RATE   == 0) ||
        (!usingRifle  && currFireRate % SHOTGUN_FIRE_RATE == 0);

    if (canFireThisFrame) {
        shooting = true;

        float ang     = sprite.getRotation() * PI / 180.f;
        sf::Vector2f dir{ std::cos(ang), std::sin(ang) };
        float        maxDist = usingRifle ? RIFLE_RANGE : SHOTGUN_RANGE;
        sf::Vector2f hitPt;
        bool         hit    = false;

        // --- 射线检测 ---
        for (int d = 1; d <= int(maxDist); ++d) {
            hitPt = sprite.getPosition() + dir * float(d);
            if (!mapManager.isPassable(hitPt)) break;
            for (auto& c : characters) {
                if (&c != this && c.getGlobalBounds().contains(hitPt)) {
                    hit = true;
                    // --- 只有这帧真正开火才扣血 ---
                    int dmg = usingRifle ? RIFLE_DAMAGE : SHOTGUN_DAMAGE;
                    c.setHealth(c.getHealth() - dmg);
                    break;
                }
            }
            if (hit) break;
        }

        // --- 记录弹道用于绘制 ---
        bulletLine[0].position = sprite.getPosition();
        bulletLine[1].position = hitPt;
        if (!usingRifle) {
            // … shotgunLine1/2 同理 …
        }
    }
// models/model_winner.pth
    // --- 无论如何，帧计数要累加 ---
    currFireRate++;
}

void Character::switchWeapon() {
    usingRifle = !usingRifle;
    if (usingRifle) {
        sprite.setTexture(rifleTexture);
        sprite.setScale(rifleScale);
    } else {
        sprite.setTexture(shotgunTexture);
        sprite.setScale(shotgunScale);
    }
}

void Character::rotate(float angleDelta) {
    sprite.setRotation(sprite.getRotation() + angleDelta);
}
void Character::draw(sf::RenderWindow& window) {
    window.draw(sprite);
    if (shooting) {
        window.draw(bulletLine);
        if (!usingRifle) {
            window.draw(shotgunLine1);
            window.draw(shotgunLine2);
        }
        if (++shootTimer>50) {
            shooting=false; shootTimer=0;
        }
    }
    drawHealthBar(window);
}

void Character::drawHealthBar(sf::RenderWindow& window) {
    const float drop=6.f, sp=2.f;
    auto pos=sprite.getPosition();
    float top = pos.y - (sprite.getLocalBounds().height*sprite.getScale().y)/2 - 10.f;
    float totalW = 10*drop + 9*sp;
    float startX = pos.x - totalW/2;
    for(int i=0;i<10;i++){
        sf::RectangleShape r({drop,drop});
        r.setPosition(startX + i*(drop+sp), top);
        r.setFillColor(i<health?sf::Color::Red:sf::Color(100,100,100));
        window.draw(r);
    }
}

std::vector<float> Character::serializeState(
    const MapManager& mapManager,
    const std::vector<Character>& characters
) const {
    std::vector<float> st;
    // 1) 自身位置归一化
    float row = sprite.getPosition().y / 32;
    float col = sprite.getPosition().x / 32;
    st.push_back(row);
    st.push_back(col);
    // 2) 对手相对位置、血量、武器、朝向
    const Character& opp = characters[1];
    float oRow = opp.sprite.getPosition().y / 32;
    float oCol = opp.sprite.getPosition().x / 32;
    st.push_back((oRow - row)/15);
    st.push_back((oCol - col)/21);
    st.push_back((float)opp.getHealth()/10.f);
    st.push_back((float)health/10.f);
    st.push_back(usingRifle?1.f:0.f);
    st.push_back(sprite.getRotation()/360.f);
    // 3) 三方向视线距离
    auto ray = [&](float delta){
        float a = sprite.getRotation()+delta;
        sf::Vector2f dir{std::cos(a*PI/180.f), std::sin(a*PI/180.f)};
        for(int d=1; d<mapManager.getWidth(); ++d){
            auto p = sprite.getPosition()+dir*float(d*32);
            if (!mapManager.isPassable(p)) return float(d)/mapManager.getWidth();
        }
        return 1.f;
    };
    st.push_back(ray(0));
    st.push_back(ray(30));
    st.push_back(ray(-30));
    assert(st.size() == 11); // 确保状态长度正确
    //std::cout << "state: " << st << std::endl;

    return st;
}

void Character::performAction(
    int action,
    const MapManager& mapManager,
    std::vector<Character>& characters
) {
    if (action<0||action>=ACTION_COUNT) return;
    switch(action){
        case 0: shoot(mapManager, characters); break;
        case 1: move({-0.2,0}, mapManager);    break;
        case 2: move({ 0.2,0}, mapManager);    break;
        case 3: move({ 0,-0.2},mapManager);    break;
        case 4: move({ 0,0.2},mapManager);    break;
        case 5: move({-0.2,-0.2},mapManager);    break;
        case 6: move({-0.2, 0.2},mapManager);    break;
        case 7: move({ 0.2,-0.2},mapManager);    break;
        case 8: move({ 0.2, 0.2},mapManager);    break;
        case 9:  rotate(-TURN_ANGLE);        break;
        case 10: rotate( TURN_ANGLE);        break;
        case 11: switchWeapon();             break;
        case 12: /* idle */                  break;
    }
}
