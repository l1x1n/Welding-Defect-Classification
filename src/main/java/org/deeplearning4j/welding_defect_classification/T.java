package org.deeplearning4j.welding_defect_classification;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class T {
    private static final Logger log = LoggerFactory.getLogger(T.class);

    public static void main(String[] args)throws Exception {

        int n = 10;
        int height = 400/n;  // height of input image
        int width = 400/n;   // width of output image
        int channels = 1; // channel
        int outputNum = 4; // 6 classes
        int batchSize = 64;
        int seed = 1234;
        Random randNumGen = new Random(seed);
        String inputDataDir = "D:/al5083/2";

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        ImageTransform cropImageTransform = new CropImageTransform(450, 150, 100, 150);


        File validData = new File(inputDataDir + "/validation");
        FileSplit validSplit = new FileSplit(validData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);


        //Testing data Vectorization
        ImageRecordReader validRR = new ImageRecordReader(height, width, channels, labelMaker);
        validRR.initialize(validSplit, cropImageTransform);
        DataSetIterator validIter = new RecordReaderDataSetIterator(validRR, batchSize, 1, outputNum);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(validIter);
        validIter.setPreProcessor(scaler);

        MultiLayerNetwork net = MultiLayerNetwork.load(new File(inputDataDir+"/4 class.zip"), true);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));




        log.info("\n*************************************** EVALUATION **********************************************\n");
        Evaluation validEva = net.evaluate(validIter);

        log.info("\n*************************************** VALIDATION EVALUATION ******************************************\n");
        log.info(validEva.stats());



    }
}
