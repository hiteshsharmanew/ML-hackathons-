import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

class_names = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

train = pd.read_csv('../input/janatahack-independence-day-2020-ml-hackathon/train.csv').fillna(' ')
test = pd.read_csv('../input/janatahack-independence-day-2020-ml-hackathon/test.csv').fillna(' ')

print(train.shape,test.shape)

train_target = train.iloc[:, 3:]
print(train_target.shape)
train_text = train['TITLE'] + ' '+ train['ABSTRACT']
test_text = test['TITLE'] + ' '+ test['ABSTRACT']
all_text = pd.concat([train_text, test_text])


def preprocess(text):
    processed_text =text.str.replace(r'\d+(\.\d+)?', 'numbr')
    # Remove punctuation
    processed_text = processed_text.str.replace(r'[^\w\d\s]', ' ')
    
    # Replace whitespace between terms with a single space
    processed_text = processed_text.str.replace(r'\s+', ' ')
    
    # Remove leading and trailing whitespace
    processed_text = processed_text.str.replace(r'^\s+|\s+?$', '')
    
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed_text = processed_text.str.lower()
    
    return processed_text

processed_text  = preprocess(all_text)


print(' word vectoriser intialisation')

    
    
word_vectorizer=TfidfVectorizer(min_df=1, smooth_idf=True, norm="l2",tokenizer=lambda x: x.split(),sublinear_tf=True, ngram_range=(1,3))    
word_vectorizer.fit(processed_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print("After vectorising this is the shape of train",train_word_features.shape)
print("After vectorising this is the shape of test",test_word_features.shape)

train_features = train_word_features
test_features = test_word_features
print('getting submission ready')
classifier=OneVsRestClassifier(LinearSVC(penalty="l2",loss='hinge',class_weight = "balanced"), n_jobs=-1)
classifier.fit(train_features, train_target)
predictions=classifier.predict(test_features)
    
## submission conversion        
submission=pd.DataFrame(predictions, columns=['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance'])
submission=pd.concat([test['ID'],submission],axis=1)
submission.to_csv("ovr_full_balanced_script_final.csv", index=False)
print(submission.shape)
submission.head()


