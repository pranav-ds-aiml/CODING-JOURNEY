import random

WORD_LISTS = {
    "easy": [
        ("python", "Programming language"),
        ("computer", "Electronic device"),
        ("keyboard", "Input device"),
        ("mouse", "Pointing device"),
        ("monitor", "Display screen")
    ],
    "medium": [
        ("algorithm", "Step-by-step procedure"),
        ("function", "Reusable code block"),
        ("variable", "Data container"),
        ("dictionary", "Key-value pairs"),
        ("programming", "Writing code")
    ],
    "hard": [
        ("encapsulation", "OOP principle"),
        ("polymorphism", "Many forms concept"),
        ("recursion", "Function calling itself"),
        ("optimization", "Making code better"),
        ("asynchronous", "Non-blocking execution")
    ]
}

HANGMAN_STAGES = [
    """
       ------
       |    |
       |
       |
       |
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |
       |
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |    |
       |
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |   /|
       |
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |   /|\\
       |
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |   /|\\
       |   /
       |
    -------
    """,
    """
       ------
       |    |
       |    O
       |   /|\\
       |   / \\
       |
    -------
    """
]

def choose_word():
    """LET PLAYER CHOOSE DIFFICULTY AND GET A WORD"""
    print("\n=== CHOOSE DIFFICULTY ===")
    print("\n=== CHOOSE DIFFICULTY ===")
    print("1. Easy (5-8 letters)")
    print("2. Medium (8-12 letters)")
    print("3. Hard (12+ letters)")
    
    choice=input("\nENETR CHOICE (1-3):")

    difficulty_map={"1":"easy","2":"medium","3":"hard"}
    difficulty=difficulty_map.get(choice,"easy")
    word,hint=random.choice(WORD_LISTS[difficulty])
    return word.upper(),hint,difficulty

def display_game_state(word,guessed_letters,wrong_guesses,hint):
    """DISPLAY CURRENT GAME STATE"""
    print("\n"+"="*50)

    print(HANGMAN_STAGES[len(wrong_guesses)])

    display=""
    for letter in word:
        if letter in guessed_letters:
            display+=letter+" "
        else:
            display+="_ "
    print(f"\nWORD: {display}")
    print(f"HINT: {hint}")
    print(f"\nWRONG GUESSES ({len(wrong_guesses)}/6): {', '.join(wrong_guesses) if wrong_guesses else 'None'}")
    print(f"Guessses letters: {', '.join(sorted(guessed_letters))}")
    print("="*50)

def play_hangman():
    """MAIN GAME LOGIC"""
    word,hint,dificulty=choose_word()
    guessed_letters=set()
    wrong_guesses=[]
    max_wrong=6

    print(f"\n STARTING GAME-DIFFICULTY: {dificulty.upper()}")
    print(f"THE WORD HAS {len(word)} letters")

    while True:
        display_game_state(word,guessed_letters,wrong_guesses,hint)
        

        if all(letter in guessed_letters for letter in word):
            print("YOU WON!")
            return True

        guess=input("\nGuess A Letter (or type 'hint' for another clue):").upper()

        if len(guess) !=1 or not guess.isalpha():
            if guess=="HINT":
                unguessed=[l for l in word if l not in guessed_letters]
                if unguessed:
                    reveal=random.choice(unguessed)
                    print(f"\n Hint: The Word Contains '{reveal}'")
                    guessed_letters.add(reveal)
                continue
        if guess in guessed_letters:
            print(" You Already Guessed That Letter")
            continue
        guessed_letters.add(guess)

        if guess in word:
            print(f"CORRECT '{guess}' is in word")
        else:
            wrong_guesses.append(guess)
            print(f"WRONG '{guess}' is not in the word")

            if len(wrong_guesses) >= max_wrong:
             display_game_state(word, guessed_letters, wrong_guesses, hint)
             print("Game Over!")
             print(f"The word was: {word}")
             return False   

def main():
    """MAIN GAME WITH SCORE TRACKING"""
    print("="*50)
    print("HANGMAN GAME")
    print("="* 50)
    print("\nGuess The Word Letter By Letter")
    print("You Have 6 Wrong Guesses Before Game Over")

    games_played=0
    games_won=0

    while True:
        won=play_hangman()

        games_played+=1
        if won:
            games_won+=1
        
        win_rate=(games_won/games_played*100) if games_played>0 else 0
        print(f"\n Your Stats: {games_won}/{games_played} wins ({win_rate:.0f}%)")
        break

if __name__=="__main__":
    main()    

        
