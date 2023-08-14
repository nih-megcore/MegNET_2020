#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:13:28 2023

@author: jstout
"""

https://keras.io/guides/customizing_what_happens_in_fit/

cv_num=0
sample = cv[cv_num]
tr, te = get_cv_npyArr(sample,
                      holdout=None,
                      arrTimeSeries=tsttr_ts,  #Subsampled array
                      arrSpatialMap=tsttr_sp, #Subsampled array
                      class_ID=tsttr_clID,  #Subsampled array
                    )

SP_, TS_ , CL_ = tr['sp'],tr['ts'], tr['clID'] 
# SP_, TS_ , CL_ = make_dual_smote_sample(tr['sp'],tr['ts'], tr['clID'], seed=int(cv_num))  
                   
history_tmp = kModel.fit(x=dict(spatial_input=SP_, temporal_input=TS_), y=CL_,
                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,   #validation_split=VALIDATION_SPLIT,
                     validation_data=(dict(spatial_input=te['sp'], temporal_input=te['ts']), te['clID']),
                     class_weight=class_weights)



from tensorflow import keras


from tensorflow import keras
# import tensorflow_addons as tfa
model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
kModel = keras.models.load_model(model_fname, compile=False)

def get_blocks(ts_current):
    num_blocks = (ts_current.shape[1] - 15000) // overlap
    blocks = np.zeros([num_blocks, 20, 15000], dtype=float)
    starts = np.arange(num_blocks)*3750
    for i,start in enumerate(starts): 
        blocks[i,:,:] = ts_current[:,start:start+15000]
    return blocks

def block_predict(sp_current, ts_current):
    '''Iterate over time to get an average prediction over time
    Returns block of 20 with average prediction value in each of 4 categories
    '''
    overlap = 3750
    blocks = get_blocks(ts_current)
    preds = np.zeros([blocks.shape[0], 20, 4]) #Time blocks / 20 components / 4 classes
    for i in range(blocks.shape[0]): 
        preds[i,:, :]=kModel.predict(dict(spatial_input=sp_current, temporal_input=blocks[i,:,:].squeeze() ))
    pred = preds.mean(axis=0)
    return pred



def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data
    
    num_subjs=len(x['temporal_input'])
    with tf.GradientTape() as tape:
        y_pred_stack = np.zeros([num_subjs, 20,4])
        overlap = 3750
        for i in range(num_subjs):
            current_in = {}
            current_in['spatial_input'] = x['spatial_input'][0+20*i:20+20*i]
            current_in['temporal_input'] = x['temporal_input'][i]
            
            blocks = get_blocks(current_in['temporal_input'])
            preds = np.zeros([blocks.shape[0], 20, 4]) #Time blocks / 20 components / 4 classes
            for i in range(blocks.shape[0]): 
                preds[i,:, :]=self(dict(spatial_input=current_in['spatial_input'],
                                                  temporal_input=blocks[i,:,:].squeeze()), y[0+20*i:20+20*i], training=True)
            print('Trying to predict')
            y_pred_stack[i,:,:] = preds.mean(axis=0) #Average over all temporal predictions - or do we want weighted
        
        y_pred = np.argmax(y_pred_stack, axis=2).reshape(-1)  #Do we want argmax or weighted 
        
        # y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

kModel.train_step = train_step.__get__(kModel)

history_tmp = kModel.fit(x=dict(spatial_input=SP_, temporal_input=TS_), y=CL_,
                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,   #validation_split=VALIDATION_SPLIT,
                     validation_data=(dict(spatial_input=te['sp'], temporal_input=te['ts']), te['clID']),
                     class_weight=class_weights)

kModel.predict(x=dict(spatial_input=SP_[0:20], temporal_input=TS_[0][:,0:15000]))




# =============================================================================
# 
# =============================================================================




# =============================================================================
# 
# =============================================================================

kModel.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(), #CategoricalCrossentropy(), 
    optimizer='Adam',
    metrics=['accuracy'] #f1mac_score]#,'accuracy']
    )

class_weights={0:1, 1:1, 2:1, 3:1}
# class_weights={0:1, 1:10, 2:10, 3:10}

history=[]
score_list=[]
tt_final = final.drop(index=holdout_dframe_idxs)
tt_final.reset_index(inplace=True, drop=True)
cv = cvSplits.main(kfolds=8, foldNormFields=crossval_cols, data_dframe=tt_final)
for cv_num in cv.keys():
    sample = cv[cv_num]
    tr, te = get_cv_npyArr(sample,
                          holdout=None,
                          arrTimeSeries=tsttr_ts,  #Subsampled array
                          arrSpatialMap=tsttr_sp, #Subsampled array
                          class_ID=tsttr_clID,  #Subsampled array
                        )
    
    SP_, TS_ , CL_ = make_dual_smote_sample(tr['sp'],tr['ts'], tr['clID'], seed=int(cv_num))  
                       
    history_tmp = kModel.fit(x=dict(spatial_input=SP_, temporal_input=TS_), y=CL_,
                         batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,   #validation_split=VALIDATION_SPLIT,
                         validation_data=(dict(spatial_input=te['sp'], temporal_input=te['ts']), te['clID']),
                         class_weight=class_weights)
    score_list.append(kModel.evaluate(x=dict(spatial_input=hold_sp, temporal_input=hold_ts), y=hold_clID))  
    history.append(history_tmp)

from MEGnet.megnet_utilities import fPredictChunkAndVoting
fPredictChunkAndVoting(self, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750)


# =============================================================================
# 
# =============================================================================

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))