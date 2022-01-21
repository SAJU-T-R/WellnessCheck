# from nltk.corpus import movie_reviews
# from nltk.classify import NaiveBayesClassifier
# from nltk.classify.util import accuracy as nltk_accuracy


# import nltk# This was ccomented
#
# nltk.download('vader_lexicon')# This was commented
# from nltk.corpus import movie_reviews

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function to print sentiments
# of the sentence.
def SentimentAnalyzer(text):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(text)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

    print("Sentence Overall Rated As", end = " ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        k = "Positive"
    elif sentiment_dict['compound'] <= - 0.05 :
    	k = "Negative"
    else :
    	k = "Neutral"

    return k


# # Driver code
# if __name__ == "__main__" :
#
# 	print("\n1st statement :")
# 	sentence = "Geeks For Geeks is the best portal for \
# 				the computer science engineering students."
#
# 	# function calling
# 	sentiment_scores(sentence)
#
# 	print("\n2nd Statement :")
# 	sentence = "study is going on as usual"
# 	sentiment_scores(sentence)
#
# 	print("\n3rd Statement :")
# 	sentence = "I am very sad today."
# 	sentiment_scores(sentence)









# Actaul code

#
# def extract_features(words):
#     return dict([(word, True) for word in words])
#
# import nltk# This was ccomented
#
# nltk.download('movie_reviews')# This was commented
# from nltk.corpus import movie_reviews
#
# def SentimentAnalyzer(text):
#     # load movie reviews from sample data
#     fileids_pos = movie_reviews.fileids('pos')
#     fileids_neg = movie_reviews.fileids('neg')
#
#     features_pos = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in fileids_pos]
#     features_neg = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in fileids_neg]
#
#     threshold = 0.8
#     num_pos = int(threshold*len(features_pos))
#     num_neg = int(threshold*len(features_neg))
#
#     # creating training and testing data
#     features_train = features_pos[:num_pos] + features_neg[:num_neg]
#     features_test = features_pos[num_pos:] + features_neg[num_neg:]
#
#     #print('\nNumber of training datapoints:', len(features_train))
#     #print('Number of test datapoints:', len(features_test))
#
#     # training a naive bayes classifier
#     classifier = NaiveBayesClassifier.train(features_train)
#     print('Accuracy:',nltk_accuracy(classifier, features_test))
#
#     probabilities = classifier.prob_classify(extract_features(text.split()))
#     # Pick the maximum value
#     predicted_sentiment = probabilities.max()
#     print("Predicted sentiment:", predicted_sentiment)
#     print("Probability:",round(probabilities.prob(predicted_sentiment), 2))
#
#     return predicted_sentiment
# # SentimentAnalyzer('It was not that good.')
