# lstm readout

previous versions of the model queried memory using the similarity of current stimulus to stored stimuli representations. the retrieved memory was a linear combination of context representations weighted by the similarity of the corresponding stimuli.

the current model will still query memory with stimulus. but instead of retrieving a linear combination weighted by stimulus similarity, this version will pass retrieved contexts through an lstm. that is, retrieved contexts will not be linearly combined, but sequentially fed through an lstm. 

NB as the number of trial increases to a large number, this implementation would suggests a large number of things are being considered. this could in the future change to involve sampling items from memory with probability given by similarity