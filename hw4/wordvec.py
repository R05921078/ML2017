import word2vec
import numpy as np
import nltk

'''
    word2vec.word2vec(
        train,
        output,
        cbow,
        size,
        min_count,
        window,
        negative,
        iter_,
        alpha,
        verbose)
'''
plot_num = 633
word2vec.word2phrase('./data/all.txt', './data/all-phrases', verbose=True)
word2vec.word2vec('./data/all-phrases', './data/words.bin', size=633, verbose=True)
model = word2vec.load('./data/words.bin')

vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:plot_num]
vocabs = vocabs[:plot_num]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=45)
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''
import matplotlib.pyplot as plt
from adjustText import adjust_text

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]


plt.figure()
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

# plt.savefig('hp.png', dpi=600)
plt.show()
