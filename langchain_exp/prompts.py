INTENT_CLASSIFIER_QUERY_GENERATION = """Create a diverse set of 50 user queries that would lead to the generation of TypeScript code snippets. Ensure the queries encompass various aspects of TypeScript programming such as class definitions, interface creations, function implementations, type aliases, enums, generics, async/await usage, and error handling. Include both beginner-level questions and advanced coding scenarios to cover a broad spectrum of TypeScript application development needs. 

Format your entries as follows:
Query: [User's Utterance] Intent: TypeScript Code Generation

Write a csv file and Continue with additional entries, each representing a unique user request that prompts the creation or explanation of TypeScript code."""


Prompt_3D_GAME = """Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:
 
 - Allow players to take turns to input their moves.
 - Check for invalid moves (e.g., placing a marker on an already occupied space).
 - Determine and announce the winner or if the game ends in a draw.
 
 Requirements:
 - Use a 2D list to represent the Tic-Tac-Toe board.
 - Use functions to modularize the code.
 - Validate player input.
 - Check for win conditions and draw conditions after each move."""

IMAGE_PREPROCESS = """
Create a python program that preprocess image for OCR image to text tasks.

- save the proccessed image to local so developers can view and check
- preprocessing technique includes
1) Binarization
2) Skew Correction
3) Noise Removal
4) Thinning and Skeletonization
- refactor the code to class and make it easy to use
- write a main function for the generated code

"""
