# onlineSG
Code repository for the Thesis work on Online Learning Security Games by Lorenzo Bisi

## If you want to see some examples
- try Example.ipynb, or view it online
    + Stackelberg(1) VS Uniform(1)

## Game steps:
- For each Player:
    + get feedback (if not in the first turn)
    + compute reward (if not in the first turn)
- For each Defender:
    + compute a strategy (if not a follower!)
- For each Attacker:
    + possibly observe defenders strategies (if follower)
    + compute a strategy
- For each player 
    + play (extract from a mixed or play a pure)


##Initialization from a configuration file (csv to be used with pandas)

- configuration file:
    + T (targets)
    + N (players)
    + payoffs (can be values or interval -> decide what to do with intervals)
    + players (always check if the payoffs matrix and the players are compatible)
        * type (A or D)
        * resources
        * profile (variable number of columns, some mandatory other optional)


- each line is a game
    + target values (game matrix)
    + attackers
    + defenders
    + followers
    + number of turns (T)

- Each Player need to know:
    + target values for all the players
    + his identity (attacker, defender)
    + his resources
