# Pronunciation_Tool
The problem we hope to address with this project is the lack of tools out there for people to practice correct pronunciation of words where the software identifies what was pronounced incorrectly and repeats the word back to the user slowly and correctly. We believe that this problem will have plenty of applications, particularly in the education system for either those with speech impediments or non-native English speakers.

## How to run on Localhost
1. Navigate to root of Pronounciation tool in the command line
2. Run "python frontend.py"
3. Fully functioning web app should appear in the browser at localhost:5000
4. Within the web app there are 3 main pages you can navigate to:
      
      a) "CMU POCKETSPHINX" which allows you to go through a 10 word evaluation to see which words you pronounce correctly
      
      b) "OUR SPECIALIZED MODEL" which brings you to a speech recognition page but that currently only evaluates the pronounciation of the word "time"
      
      c) "ADD TRAINING DATA" which looks very similar to the last page but it was added for us to be able to quickly and accurately add data to train and test our tools.

## How to run the test to compare accuracy
1. Navigate to root of Pronounciation tool in the command line
2. Run "python test.py"
3. There will be 4 meaningful outputs from this program:
      a) "CMU Accuracy" followed by a percentage of accurate predictions in the train data directory by the CMU PocketSphinx tool
      b) "Naive Bayes Accuracy" and the percentage of accurate predictions with the validation data
      c) "Decision Tree Accuracy" is like the last one but with the model being trained being a Decision Tree Classifier
      d) "Max Entropy Accuracy" is again like the last two but using the Maximum Entropy Classifier
