import math
from oddsProb import *

#dictionary of all unique vocabulary words
total_vocab = dict([])

#dictionay of unique words belonging to "positive" class (labeled +1)
positive_dict = dict([])
#total count of words, including duplicates, belonging to the positive dictionary
total_pos = 0

#dictionay of unique words belonging to "negative" class (labeled -1)
negative_dict = dict([])
#total count of words, including duplicates, belonging to the negative dictionary
total_neg = 0

#true positives
both_pos = 0

#false negatives
std_only_pos_test_neg = 0

#false positives
std_only_neg_test_pos = 0

#true negatives
both_neg = 0

#macros for the training file and testing file
TRAINING_FILE = 'rt-train.txt'
TESTING_FILE = 'rt-test.txt'



"""
Laplace smoothing function
num = frequency of a particular word in the particular dictionary
denom = total number of words in that same dictionary
len(total_vocab) = total number of unique words in the overall vocabulary
"""

def smooth(num, denom):
    my_num = (float)(num+1)
    my_denom = (float)(denom+len(total_vocab))
    return my_num/my_denom


#implement training
with open(TRAINING_FILE, "r") as train_file:
    for line in train_file:
        #split each line up
        word_split = line.split()
        index = 0
        
        for word in word_split:
            if index > 0:
                #parse each individual word and its frequency
                filtered = word.strip('\n')
                split_position = filtered.find(":")
                current_word = filtered[:split_position]
                current_stat = (int)(filtered[split_position+1:])
                
                #if not in universal dictionary, add it to the universal dictionary
                if(total_vocab.get(current_word) == None):
                    total_vocab[current_word] = True
                
                """
                if document is labeled positive, check each word to see if in positive dictionary
                if not, then add it to the positive dictionary and hash the frequency to that word label
                if it is, then increment the frequency that is existing by the amount parsed
                """
                if ((int)(word_split[0]) == 1):
                    if positive_dict.get(current_word) == None:
                        positive_dict[current_word] = current_stat
                    else:
                        positive_dict[current_word] += current_stat
                    total_pos += current_stat
                elif ((int)(word_split[0]) == -1):                
                    """    
                    if document is labeled positive, check each word to see if in positive dictionary
                    if not, then add it to the positive dictionary and hash the frequency to that word label
                    if it is, then increment the frequency that is existing by the amount parsed    
                    """
                    if negative_dict.get(current_word) == None:
                        negative_dict[current_word] = current_stat
                    else:
                        negative_dict[current_word] += current_stat
                    total_neg += current_stat
            
            index += 1




#now with the testing
with open(TESTING_FILE, "r") as test_file:
    #parse each line by using the split() function
    for line in test_file:
        word_split = line.split()
        index = 0
        
        #take the first element and set it as your "correct" class label
        correct_std = (int)(word_split[0])
        
        #initialize your test assessment class label
        guessed_std = 0
        
        
        """
        set your priors for the class frequency,
        since we have 1000 positive documents, 1000 negative documents, 2000 total documents in training sets
        the priors are calculated as follow:
        
        "document of particular class"/total_documents
        
        """
        pos_prob = math.log(float(1000)/2000, 2)
        neg_prob = math.log(float(1000)/2000,2)
        
        for word in word_split:
            if index > 0:
                #parse the words and frequency from test document
                filtered = word.strip('\n')
                split_position = filtered.find(":")
                current_word = filtered[:split_position]
                current_stat = (int)(filtered[split_position+1:])
                
                
                """
                if word is already in positive dictionary, get the frequency from the dictionary,
                smooth it using the function, and update the probability
                
                else, substitute 0 as the frequency, use the smooth function, and update the probability
                """
                if(positive_dict.get(current_word) != None):
                    single_prob = smooth(positive_dict.get(current_word), total_pos)
                    pos_prob += math.log(math.pow(single_prob, current_stat), 2)
                else:
                    pos_prob += math.log(smooth(0, total_pos), 2)
                
                """
                if word is already in negative dictionary, get the frequency from the dictionary,
                smooth it using the function, and update the probability
                
                else, substitute 0 as the frequency, use the smooth function, and update the probability
                """
                if(negative_dict.get(current_word) != None):
                    single_prob = smooth(negative_dict.get(current_word), total_neg)
                    neg_prob += math.log(math.pow(single_prob, current_stat), 2)
                else:
                    neg_prob += math.log(smooth(0, total_neg), 2)
            
            index += 1
            
        """
        if the positive probability per Bayes' rule is greater than negative,
        evaluate the document as "positive"
        else, evaluate as negative
        """
        if pos_prob > neg_prob:
            guessed_std = 1
        else:
            guessed_std = -1

        if(correct_std == 1 and guessed_std == 1):
            #if standard is + and eval is +, increment "true positive"
            both_pos += 1
        elif(correct_std == -1 and guessed_std == -1):
            #if standard is - and eval is -, increment "true negative"
            both_neg += 1
        elif(correct_std == 1 and guessed_std == -1):
            #if standard is + and eval is -, increment "false negative"
            std_only_pos_test_neg += 1
        elif(correct_std == -1 and guessed_std == 1):
            std_only_neg_test_pos += 1



print "vocab size " + str(len(total_vocab))
print "pos length " + str(total_neg)

print("\n\n")
total = both_pos + both_neg + std_only_pos_test_neg+ std_only_neg_test_pos

print "True Positive " + str(both_pos)
print "True Negative " + str(both_neg)  
print "False Negative " + str(std_only_pos_test_neg) 
print "False Positive " + str(std_only_neg_test_pos)

print "Positive classification rate:  " +  str(float(both_pos+std_only_neg_test_pos)/total)
print "negative classification rate:  " + str(float(both_neg+std_only_pos_test_neg)/total)
print "Accurate Classification rate: " + str(float(both_neg+both_pos)/total)


my_array = []


"""
ODDS RATIO CALCULATIONS
"""

#iterate through positive dictionary, this is your target reference dictionary for now
for key in positive_dict:
    num = 1
    denom = 1
    
    #for each existing word, get the frequency in current dictionary
    num = smooth(positive_dict.get(key), total_pos)
    """
    now look for the same word in the negative dictionary
    if exists, use the frequency listed in dictionary there with the smooth function
    if not exists, then use 0 as your frequency for your smoothing function
    """
    if(negative_dict.get(key) == None):
        denom = smooth(0, total_neg)
    else:
        denom = smooth(negative_dict.get(key), total_neg)
    
    """
    place the odds calculation into an object that tracks it for a given word in a given dictionary
    num = likelihood in current target dictionary
    denom = likelihood in opposite dictionary
    key = word you are looking for in reference dictionary
    """
    temp = oddsProb(key, num/denom)
    my_array.append(temp)
    
sorted_list = sorted(my_array, key=lambda x: x.my_prob, reverse=True)

print("\n\n\nPOSITIVE ODDS")
index = 0
for item in sorted_list:
    print item.my_word, item.my_prob
    index +=1
    if index == 10:
        break 


#iterate through negative dictionary, this is your current reference dictionary
my_array = []
index = 0
for key in negative_dict:
    num = 1
    denom = 1
    
    
    #for each existing word, get the frequency in current dictionary
    num = smooth(negative_dict.get(key), total_neg)
    
    """
    now look for the same word in the positive dictionary
    if exists, use the frequency listed in dictionary there with the smooth function
    if not exists, then use 0 as your frequency for your smoothing function
    """
    if(positive_dict.get(key) == None):
        denom = smooth(0, total_pos)
    else:
        denom = smooth(positive_dict.get(key), total_pos)
    
    """
    place the odds calculation into an object that tracks it for a given word in a given dictionary
    num = likelihood in current target dictionary
    denom = likelihood in opposite dictionary
    key = word you are looking for in reference dictionary
    """
    temp = oddsProb(key, num/denom)
    my_array.append(temp)
    
sorted_list = sorted(my_array, key=lambda x: x.my_prob, reverse=True)

print ("\n\nNEGATIVE ODDS")
for item in sorted_list:
    print item.my_word, item.my_prob
    index +=1
    if index == 10:
        break 

 

"""
LIKELIHOOD CALCULATIONS
"""

#sort frequency of items in negative dictionary, since all items have the same denominator ignore the denominator
sort_x = sorted(negative_dict.iteritems(), key=lambda (k,v): (v,k), reverse = True)

#only print top 10
print("\n\n\nLIKELIHOOD POSITIVE")
index = 0
for item in sort_x:
    print item[0]
    index += 1
    if index == 10:
        break


#sort frequency of items in positive dictionary, since all items have the same denominator ignore the denominator    
sort_x = sorted(positive_dict.iteritems(), key=lambda (k,v): (v,k), reverse = True)

#only print top 10
print("\n\n\nLIKELIHOOD NEGATIVE")
index = 0
for item in sort_x:
    print item[0]
    index += 1
    if index == 10:
        break
    

