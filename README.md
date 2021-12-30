# parallel-text-generation

This project generates clickbait headlines in parallel with multiple processors. A markov chain model is trained using input data from clickbait_data.txt. 
The program then runs like a game of telephone, with each processor generating the next word based on the predictions from the markov chain, and then passing the headline to the next processor. 
The final result is then printed out. Message Passing Interface (MPI) is used to run the program in parallel.
