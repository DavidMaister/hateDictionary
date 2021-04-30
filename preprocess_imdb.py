import os
import io

# file to store the reviews line by line
# opening and writing files with this syntax to encode with utf-8 and not getting ascii errors
with io.open('imdb_reviews.txt', 'w', encoding='utf8') as f:
    counter = 0
    for review in os.listdir('aclImdb/train/pos'):
        with io.open('aclImdb/train/pos/' + review, 'r', encoding='utf8') as f_review:
        # open the file  with the review
            text = f_review.read()  # file only contain a single line with the review
            f.write(text + '\n')    # write a line in the final file with the review
        #f_review.close()
        if counter > 4000:
            break
        counter += 1

    counter = 0
    for review in os.listdir('aclImdb/train/neg'):  # same for for positive than wth negative
        with io.open('aclImdb/train/neg/' + review, 'r', encoding='utf8') as f_review:
            # open the file  with the review
            text = f_review.read()  # file only contain a single line with the review
            f.write(text + '\n')  # write a line in the final file with the review
        #f_review.close()
        if counter > 4000:
            break
        counter += 1



