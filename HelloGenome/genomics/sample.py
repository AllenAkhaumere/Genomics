
import sys
sys.path.append('../')
import genomics.utils as utils
import genomics.genome as gene

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from  matplotlib import cm

Xarray, Yarray, pos_array = \
utils.get_training_array("C:/Medical Dataset/Dataset/variant_data/aln_tensor_chr21",
                         "C:/Medical Dataset/Dataset/variant_data/variants_chr21",
                         "C:/Medical Dataset/Dataset/variant_data/testing_data/chr21/CHROM21_v.3.3.2_highconf_noinconsistent.bed" )
print(Xarray.shape)
print(Yarray.shape)

i = 999
pl.figure(figsize=(5, 2))
plt.matshow(Xarray[i,:,:,0].transpose(), vmin=0, vmax=50, cmap= cm.coolwarm, fignum=0)
pl.figure(figsize=(5, 2))
plt.matshow(Xarray[i,:,:,1].transpose(), vmin=-50, vmax=50, cmap=cm.coolwarm, fignum=0)
pl.figure(figsize=(5, 2))
plt.matshow(Xarray[i,:,:,2].transpose(), vmin=-50, vmax=50, cmap=cm.coolwarm, fignum=0)

gen = gene.GeneNet()
gen.init()

batch_size = 500
validation_lost = []
for i in range(2401):
    Xbatch, Ybatch = utils.get_batch(Xarray[:30000], Yarray[:30000], size=batch_size)
    loss = gen.train(Xbatch, Ybatch)
    if i % (len(Xarray[:30000])/batch_size) == 0:
        v_lost = gen.get_loss( Xarray[30000:40000], Yarray[30000:40000] )
        print(i, "train lost:", loss/batch_size, "validation lost", v_lost/10000)
        gen.save_parameters('C:/Medical Dataset/Dataset/variant_data/parameters/vn.params-%04d' % i)
        validation_lost.append( (v_lost, i) )

        # pick the parameter set of the smallest validation loss

        validation_lost.sort()
        i = validation_lost[0][1]
        print
        i
        gen.restore_parameters('C:/Medical Dataset/Dataset/variant_data/parameters/vn.params-%04d' % i)
    Xarray2, Yarray2, pos_array2 = \
        utils.get_training_array("C:/Medical Dataset/Dataset/variant_data/aln_tensor_chr22",
                                 "C:/Medical Dataset/Dataset/variant_data/variants_chr22",
                                 "C:/Medical Dataset/Dataset/variant_data/testing_data/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed")
    base, t = gen.predict(Xarray2)

    # we can compare the output of the expected calls and the predicted calls

    pl.figure(figsize=(15, 5))
    plt.matshow(Yarray2[4000:4150, :].transpose(), fignum=0)
    pl.figure(figsize=(15, 5))
    plt.matshow(np.concatenate((base[4000:4150, :], t[4000:4150, :]), 1).transpose(), fignum=0)

    evaluation_data = []
    for pos, predict_v, annotate_v in zip(np.array(pos_array2), t, Yarray2[:, 4:]):
        evaluation_data.append((pos, np.argmax(predict_v), np.argmax(annotate_v)))
    evaluation_data = np.array(evaluation_data)