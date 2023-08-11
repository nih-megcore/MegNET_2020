import os
import sys
import copy
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from scipy.io import loadmat
import glob
# sys.path.append('/home/jstout/src/MEGnet')
import scipy.stats as stats
#import paths
import json
import pickle
from sklearn.preprocessing import MinMaxScaler

import os.path as multi_stratified_k_fold  #Hack to get this to load totally nonsense import

def fLoadData(strRatersLabels, strDBRoot, bCropSpatial=True, bAsSensorCap=False):
    #strRatersLabels = '/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/AlexMcGillRatings.xlsx'
    #strDBRoot = '/archive/bioinformatics/DLLab/AlexTreacher/data/brainstorm_db/McGill'

    dfRatersLabels = pd.read_excel(strRatersLabels)
    intNICAComp = dfRatersLabels.columns[-1]

    lSpatial = []
    lTemporal = []
    lLabel = []
    lPath = []
    lSubject = []
    lScanType = []

    i = 0
    row = dfRatersLabels.iloc[i]
    for i, row in dfRatersLabels.iterrows():
        strDataPath = row['strPath']
        if bAsSensorCap:
            strDataPathSpatial = os.path.join(os.path.dirname(strDataPath), os.path.basename(strDataPath).replace('Disc','SensorCap'))
        else:
            strDataPathSpatial = strDataPath
        arrTemporalAll = loadmat(os.path.join(strDataPath, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
        intComp = 1
        for intComp in range(1,intNICAComp+1):
            lLabel.append(row[intComp])
            lTemporal.append(arrTemporalAll[intComp-1]) #minus one because zero index, the 1st comp is at 0 index, the 2nd at 1 index etc.
            if bCropSpatial:
                lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'][30:-30,15:-14,:])
            else:
                lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'])
            lPath.append(row['strPath'])
            lSubject.append(row['strSubject'])
            lScanType.append(row['strType'])
        if i%20 == 0:
            print(f"Loading subject {i} of {dfRatersLabels.shape[0]}")
    return lSpatial, lTemporal, lLabel, lSubject, lPath, lScanType

def fLoadAllData(strDataRoot):
    pass

def fGetStartTimesOverlap(intInputLen, intModelLen=15000, intOverlap=3750):
    """
    model len is 60 seconds at 250Hz = 15000
    overlap len is 15 seconds at 250Hz = 3750
    """
    lStartTimes = []
    intStartTime = 0
    while intStartTime+intModelLen<=intInputLen:
        lStartTimes.append(intStartTime)
        intStartTime = intStartTime+intModelLen-intOverlap
    return lStartTimes

def fChunkData(arrSpatialMap, arrTimeSeries, intLabel, intModelLen=15000, intOverlap=3750):
    intInputLen = arrTimeSeries.shape[0]
    lStartTimes = fGetStartTimesOverlap(intInputLen, intModelLen, intOverlap)

    lTemporalSubjectSlices = [arrTimeSeries[intStartTime:intStartTime+intModelLen] for intStartTime in lStartTimes]
    lSpatialSubjectSlices = [arrSpatialMap for intStartTime in lStartTimes]
    lLabel = [intLabel for intStartTime in lStartTimes]

    return lSpatialSubjectSlices, lTemporalSubjectSlices, lLabel


def fLoadAndPickleData(strTrainDFPath = None, #os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TrainScans.csv'),
                       strValDFPath = None, #os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/ValidationScans.csv'),
                       strTestDFPath = None, #os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TestScans.csv'),
                       dctDatabaseRatingPaths = {
                            'HCP_MEG_20210212':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_HCP_MEG_20210212_final.xlsx',
                            'iTAKL':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_iTAKL_final.xlsx',
                            'McGill':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_McGill_final.xlsx',
                        },
                        intNComponents = 20,
                        strDataDir = None, #os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters'),
                        strOutDir = None, #os.path.join(paths.strDataDir, 'MEGnet/FoldData'),
                        intModelLen = 60*250,
                        intOverlap=0,#for the training/validation there is no overlap set, for testing the complete lenght model (with voting) we will have a 15 second overlap
                        ):
    """

    Test data will not be chunked

    :param fortestingthecompletelenghtmodel: [description]
    :type fortestingthecompletelenghtmodel: [type]
    :param strTrainDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TrainScans.csv')
    :type strTrainDFPath: [type], optional
    :param strValDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/ValidationScans.csv')
    :type strValDFPath: [type], optional
    :param strTestDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TestScans.csv')
    :type strTestDFPath: [type], optional
    :param dctDatabaseRatingPaths: [description], defaults to { 'HCP_MEG_20210212':os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/RaterTemplateHCP_MEG_20210212_AT.csv'), 'iTAKL':os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/RaterTemplateiTAKL.csv'), 'McGill':os.path.join(paths.strDataDir, 'MEGnet/AlexMcGillRatings.csv') }
    :type dctDatabaseRatingPaths: dict, optional
    :param intNComponents: [description], defaults to 20
    :type intNComponents: int, optional
    :param strDataDir: [description], defaults to os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters')
    :type strDataDir: [type], optional
    :param strOutDir: [description], defaults to os.path.join(paths.strDataDir, 'MEGnet/FoldData')
    :type strOutDir: [type], optional
    :param intModelLen: [description], defaults to 60*250
    :type intModelLen: [type], optional
    :param intOverlap: [description], defaults to 0
    :type intOverlap: int, optional
    """
    dctLabelConv = {'N':0, 'n':0, 0:0,
                    'B':1, 'b':1, 1:1,
                    'C':2, 'c':2, 2:2,
                    'S':3, 's':3, 3:3,
                    }
    
    dfTrainScans = pd.read_csv(strTrainDFPath, index_col=0, header=[0,1])
    dfValScans = pd.read_csv(strValDFPath, index_col=0, header=[0,1])
    dfTestScans = pd.read_csv(strTestDFPath, index_col=0)

    dctDataBaseRatings = dict([(strDB, pd.read_excel(strPath, index_col=0)) for strDB, strPath in dctDatabaseRatingPaths.items()])
    
    lTestTimeSeries = []
    lTestSpatialMap = []
    lTestLabel = []

    #save the test data as is (different lenghts)    
    for i, row in dfTestScans.iterrows():
        dfRatings = dctDataBaseRatings[row['database']]
        dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
        assert dfRatingsScan.shape[0] == 1
        pdsRatingsScan = dfRatingsScan.iloc[0]
        arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
        for intComp in range(1,intNComponents+1):
            arrSptatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
            lTestTimeSeries.append(arrTimeSeries[:,intComp-1])#min one as zero indexed
            lTestSpatialMap.append(arrSptatialMap)
            lTestLabel.append(dctLabelConv[pdsRatingsScan.loc[intComp]])
    for strPath, arr in zip(['lTestTimeSeries.pkl','lTestSpatialMap.pkl','lTestY.pkl'],
                            [lTestTimeSeries,lTestSpatialMap,lTestLabel]):
        #np.save(os.path.join(strOutDir, strPath), np.stack(arr))
        with open(os.path.join(strOutDir, strPath), 'wb') as f:
            pickle.dump(arr, f)

    #save the val data in 60s chunks
    for intFold in range(np.max([int(x[0]) for x in dfValScans.columns])+1):
        lTimeSeries = []
        lSpatialMap = []
        lLabels = []
        dfVal = dfValScans[str(intFold)].dropna()
        for i, row in dfVal.iterrows():
            dfRatings = dctDataBaseRatings[row['database']]
            dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
            assert dfRatingsScan.shape[0] == 1
            pdsRatingsScan = dfRatingsScan.iloc[0]
            arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
            for intComp in range(1,intNComponents+1):
                arrSpatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
                #Split the temporal signal into 60 second chunks with 0 seconds overlap
                lSpatialSubjectSlices, lTemporalSubjectSlices, lLabelSubject = fChunkData(arrSpatialMap, 
                                                                                          arrTimeSeries[:,intComp-1], 
                                                                                          dctLabelConv[pdsRatingsScan.loc[intComp]],
                                                                                          intModelLen=intModelLen,
                                                                                          intOverlap=intOverlap)
                lTimeSeries.extend(lTemporalSubjectSlices)
                lSpatialMap.extend(lSpatialSubjectSlices)
                lLabels.extend(lLabelSubject)
        for strPath, arr in zip([f'arrValidation{intFold}TimeSeries.npy',f'arrValidation{intFold}SpatialMap.npy',f'arrValidation{intFold}Y.npy'],
                                [lTimeSeries,lSpatialMap,lLabels]):
            np.save(os.path.join(strOutDir, strPath), np.stack(arr))
            #with open(os.path.join(strOutDir, strPath), 'wb') as f:
            #    pickle.dump(arr, f)
        
    #save the train data in 60s chunks
    for intFold in range(np.max([int(x[0]) for x in dfTrainScans.columns])+1):
        lTimeSeries = []
        lSpatialMap = []
        lLabels = []
        dfTrain = dfTrainScans[str(intFold)].dropna()
        for i, row in dfTrain.iterrows():
            dfRatings = dctDataBaseRatings[row['database']]
            dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
            assert dfRatingsScan.shape[0] == 1
            pdsRatingsScan = dfRatingsScan.iloc[0]
            arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
            for intComp in range(1,intNComponents+1):
                arrSpatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
                #Split the temporal signal into 60 second chunks with 0 seconds overlap
                lSpatialSubjectSlices, lTemporalSubjectSlices, lLabelSubject = fChunkData(arrSpatialMap, 
                                                                                          arrTimeSeries[:,intComp-1], 
                                                                                          dctLabelConv[pdsRatingsScan.loc[intComp]],
                                                                                          intModelLen=intModelLen,
                                                                                          intOverlap=intOverlap)
                lTimeSeries.extend(lTemporalSubjectSlices)
                lSpatialMap.extend(lSpatialSubjectSlices)
                lLabels.extend(lLabelSubject)
        for strPath, arr in zip([f'arrTrain{intFold}TimeSeries.npy',f'arrTrain{intFold}SpatialMap.npy',f'arrTrain{intFold}Y.npy'],
                                [lTimeSeries,lSpatialMap,lLabels]):
            np.save(os.path.join(strOutDir, strPath), np.stack(arr))
            #with open(os.path.join(strOutDir, strPath), 'wb') as f:
            #    pickle.dump(arr, f)

def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750):
    """
    This function is designed to take in ICA time series and a spatial map pair and produce a prediction useing a trained model.
    The time series will be split into multiple chunks and the final prediction will be a weighted vote of each time chunk.
    The weight for the voting will be determined by the manout of time and overlap each chunk has with one another.
    For example if the total lenght of the scan is 50 seconds, and the chunks are 15 seconds long with a 5 second overlap:
        The first chunk will be the only chunk to use the first 10 seconds, and one of two chunks to use the next 5 seconds.
            Thus   

    :param kModel: The model that will be used for the predictions on each chunk. It should have two inputs the spatial map and time series respectivley
    :type kModel: a keras model
    :param lTimeSeries: The time series for each scan (can also be an array if all scans are the same lenght)
    :type lTimeSeries: list or array (if each scan is a different length, then it needs to be a list)
    :param arrSpatialMap: The spatial maps (one per scan)
    :type arrSpatialMap: numpy array
    :param intModelLen: The lenght of the time series in the model, defaults to 15000
    :type intModelLen: int, optional
    :param intOverlap: The lenght of the overlap between scans, defaults to 3750
    :type intOverlap: int, optional
    """
    #empty list to hold the prediction for each component pair
    lPredictionsVote = []
    lGTVote = []

    lPredictionsChunk = []
    lGTChunk = []

    i = 0
    for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        intTimeSeriesLen = arrScanTimeSeries.shape[0]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1]+intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0]-intModelLen)


        lTimeChunks = [[x,x+intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x,0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x+intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime]+=1.0/intInChunks

        #predict
        dctWeightedPredictions = {}
        for intStartTime in dctTimeChunkVotes.keys():
            testTimeSeries = copy.deepcopy(arrScanTimeSeries[intStartTime:intStartTime+intModelLen])
            min_vals = np.min(testTimeSeries)#, axis=1, keepdims=True)
            max_vals = np.max(testTimeSeries)#, axis=1, keepdims=True)
            scaling_factors = 10 / (max_vals - min_vals)
            mean_vals = np.mean(testTimeSeries)#, axis=1, keepdims=True)
            testTimeSeries = testTimeSeries - mean_vals
            testTimeSeries = testTimeSeries * scaling_factors 
            
            lPrediction = kModel.predict([np.expand_dims(arrScanSpatialMap,0),
                                        np.expand_dims(np.expand_dims(testTimeSeries,0),-1)])
            lPredictionsChunk.append(lPrediction)
            lGTChunk.append(arrScanY)
            
            dctWeightedPredictions[intStartTime] = lPrediction*dctTimeChunkVotes[intStartTime]

        arrScanPrediction = np.stack(dctWeightedPredictions.values())
        arrScanPrediction = arrScanPrediction.mean(axis=0)
        arrScanPrediction = arrScanPrediction/arrScanPrediction.sum()
        lPredictionsVote.append(arrScanPrediction)
        lGTVote.append(arrScanY)
        
        #print(f"{i}/{arrY.shape[0]}")
        i+=1
    return np.stack(lPredictionsVote), np.stack(lGTVote), np.stack(lPredictionsChunk), np.stack(lGTChunk)

def fReadConfig(strPath):
    """Reads a config from a text file and returns the config dict

    :param strPath: [description]
    :type strPath: [type]
    :return: [description]
    :rtype: [type]
    """
    if os.path.exists(strPath):
        with open(strPath, 'r') as f:
            dctConfig = json.loads(f.readlines()[0])
    else:
        dctConfig = None
    return dctConfig

def fGetModelStatsFromTrainingHistory(strPath):
    """Get the summary stats from the training history across folds
    Input dataframe should have a multiindex for columns. 
        0. The fold
        1. The performance metric

    :param strPath: path to history dataframe
    :type strPath: str
    :return: summary values across folds
    :rtype: dct
    """
    dfHistory = pd.read_csv(strPath, index_col=0, header=[0,1])
    dfValF1 = dfHistory.loc[:, idx[:,'val_f1_score']]
    dctReturn = {'mean_val_f1': dfValF1.max().mean(),
                 'std_val_f1': dfValF1.max().std(),
                 'mean_epochs': dfValF1.idxmax().mean()}
    return pd.Series(dctReturn)

def fSummaryDFFromRS(strRSRoot):
    """Creates a summary dataframe from the RS that includes configs and performance.
    Useful to determine best model, and for HPO analysis

    :param strRSRoot: path to the root folder that contains the RS models
    :type strRSRoot: str
    :return: pandas dataframe of the RS summary
    :rtype: pd.DataFrame
    """
    lTrainingHistoryPaths = glob.glob(os.path.join(strRSRoot,'*','training_history.csv'))
    lTrainingHistoryPaths.sort(key = lambda x: int(x.split(os.sep)[-2][5:]))
    lDFTrainingHistory = [pd.read_csv(x, index_col=0, header=[0,1]) for x in lTrainingHistoryPaths]
    lF1Val = [x.loc[:, idx[:,'val_f1_score']] for x in lDFTrainingHistory]
    dfSummary = pd.DataFrame(columns = ['model_path','model_num','mean_val_f1','std_val_f1','mean_epochs'])
    dfSummary['model_path'] = [os.path.dirname(x) for x in lTrainingHistoryPaths]
    dfSummary['model_num'] = dfSummary['model_path'].apply(lambda x: x.split(os.sep)[-1])
    dfSummary['config'] = [fReadConfig(os.path.join(x, 'config.txt')) for x in dfSummary['model_path']]

    dfSummary[['mean_val_f1','std_val_f1','mean_epochs']] = (dfSummary['model_path']+'/training_history.csv').apply(fGetModelStatsFromTrainingHistory)
    dfSummary['95CI_val_f1'] = dfSummary['mean_val_f1']-1.96*dfSummary['std_val_f1']
    return dfSummary

def fPredictChunkAndVoting_parrallel(kModel, lTimeSeries, arrSpatialMap, intModelLen=15000, intOverlap=3750):
    """
    This function is designed to take in ICA time series and a spatial map pair and produce a prediction useing a trained model.
    The time series will be split into multiple chunks and the final prediction will be a weighted vote of each time chunk.
    The weight for the voting will be determined by the manout of time and overlap each chunk has with one another.
    For example if the total lenght of the scan is 50 seconds, and the chunks are 15 seconds long with a 5 second overlap:
        The first chunk will be the only chunk to use the first 10 seconds, and one of two chunks to use the next 5 seconds.
            Thus   

    :param kModel: The model that will be used for the predictions on each chunk. It should have two inputs the spatial map and time series respectivley
    :type kModel: a keras model
    :param lTimeSeries: The time series for each scan (can also be an array if all scans are the same lenght)
    :type lTimeSeries: list or array (if each scan is a different length, then it needs to be a list)
    :param arrSpatialMap: The spatial maps (one per scan)
    :type arrSpatialMap: numpy array
    :param intModelLen: The lenght of the time series in the model, defaults to 15000
    :type intModelLen: int, optional
    :param intOverlap: The lenght of the overlap between scans, defaults to 3750
    :type intOverlap: int, optional
    """
    #empty list to hold the prediction for each component pair
    lPredictionsVote = []
    lGTVote = []

    lPredictionsChunk = []
    lGTChunk = []
    allChunkPredictions = []
    
    # if arrY==None:
    #     arrY=np.zeros(len(arrSpatialMap))

    i = 0
    num_subjs = arrSpatialMap.shape[0]//20
    arrSpatialMap = arrSpatialMap.reshape(arrSpatialMap.shape[0]//20, 20, 120,120,3)
    lTimeSeries = lTimeSeries.reshape(lTimeSeries.shape[0]//20, 20, -1)
    
    for subj_idx in range(num_subjs):
    # for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        lPredictionsChunk=[]
        arrScanTimeSeries=lTimeSeries[subj_idx, :,:]
        arrScanSpatialMap = arrSpatialMap[subj_idx,:,:,:,:]
        intTimeSeriesLen = lTimeSeries.shape[-1]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1]+intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0]-intModelLen)
            
        lTimeChunks = [[x,x+intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x,0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x+intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime]+=1.0/intInChunks

        #predict
        dctWeightedPredictions = {}
        dctTimeChunkVotes = {i:j for i,j in dctTimeChunkVotes.items() if i>0}
        for intStartTime in dctTimeChunkVotes.keys():
            testTimeSeries = copy.deepcopy(arrScanTimeSeries[:, intStartTime:intStartTime+intModelLen])
            min_vals = np.min(testTimeSeries, axis=1, keepdims=True)
            max_vals = np.max(testTimeSeries, axis=1, keepdims=True)
            scaling_factors = 8 / (max_vals - min_vals)
            mean_vals = np.mean(testTimeSeries, axis=1, keepdims=True)
            testTimeSeries = testTimeSeries - mean_vals
            testTimeSeries = testTimeSeries * scaling_factors 
            
            lPrediction = kModel.predict([arrScanSpatialMap,testTimeSeries])
            lPredictionsChunk.append(lPrediction)
            
            dctWeightedPredictions[intStartTime] = lPrediction*dctTimeChunkVotes[intStartTime]

        arrScanPrediction = np.stack(dctWeightedPredictions.values())
        arrScanPrediction = arrScanPrediction.mean(axis=0)
        arrScanPrediction = arrScanPrediction/arrScanPrediction.sum()
        lPredictionsVote.append(arrScanPrediction)
        allChunkPredictions.append(np.stack(lPredictionsChunk, axis=-1))
        i+=1
    lPredictionsVote = np.stack(lPredictionsVote)
    lPredictionsVote = lPredictionsVote.reshape(lPredictionsVote.shape[0]*lPredictionsVote.shape[1],-1)
    return lPredictionsVote , np.stack(allChunkPredictions) #, np.stack(lGTChunk)

