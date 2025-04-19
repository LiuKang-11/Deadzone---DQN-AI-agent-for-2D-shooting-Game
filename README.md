# Deadzone
This game is developed as part of the CSC 584 Spring 2025 course project.

# Introduction
We implement a DQN Model on Top-Down Shooter game, and figure out optimal rewards systems and the behaviors they promote.

# Game design
- Demo Video:
https://www.youtube.com/watch?v=FVIRmR6Go_8&ab_channel=%E5%8A%89%E6%A2%93%E5%A0%82
![image](https://github.com/LiuKang-11/Deadzone/blob/main/screenshot/image1.png)
![image](https://github.com/LiuKang-11/Deadzone/blob/main/screenshot/image2.png)
![image](https://github.com/LiuKang-11/Deadzone/blob/main/screenshot/image3.png)


## States:
- Player Position
- Relative Opponent Position
- Relative Obstacle Positions
- Weapon Type
- Player Health
- Opponent Health
- Current Orientation

## Action
- Move
- Turn
- Shoot
- Switch
- Idle


# How to test

- Run ai_server.py to open the socket:
```
python ai_server.py
```
- Then run the main excutable file:
```
./main
```

# For training
Train.py provide a platform to design your own reward function and training your AI agent, once the model's weight is saved, you can load it in ai_server.py

# Collaborators
- Hrishikesh Salway
- Leslie Liu
- Akhilesh Neeruganti
