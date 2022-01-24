package org.deeplearning4j.welding_defect_classification;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Random;

public class WeldingDefectClassification {
    private static final Logger log = LoggerFactory.getLogger(WeldingDefectClassification.class);

    public static void main(String[] args) throws Exception {
        int height = 40;  // height of input image
        int width = 40;   // width of output image
        int channels = 1; // channel
        int outputNum = 4; // 4 classes
        int batchSize = 32;
        int nEpochs = 3;
        int seed = 1234;
        Random randNumGen = new Random(seed);
        String inputDataDir = "D:/al5083/2";


        File trainData = new File(inputDataDir + "/train"); // loading images
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // generate labels. parent path as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        ImageTransform cropImageTransform = new CropImageTransform(450, 150, 100, 150);

        trainRR.initialize(trainSplit, cropImageTransform);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

// normalization of grayscale image from 0-255 to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter); //normalize images
        trainIter.setPreProcessor(scaler);

//Testing data Vectorization
        File testData = new File(inputDataDir + "/test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit, cropImageTransform);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(scaler); // same normalization for better results

// NN configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // create a NN
                .seed(seed)
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(5e-4))//5e-4
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .name("Conv1")
                        .nIn(channels)
                        .nOut(16) //16
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .name("Pooling1")
                        .poolingType(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .name("Conv2")
                        .nOut(16)  //16
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .name("Pooling2")
                        .poolingType(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .name("Conv3")
                        .nOut(32) //32
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .name("Pooling3")
                        .poolingType(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(new DenseLayer.Builder()
                        .name("dense1")
                        .nOut(32)  //32
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .name("dense2")
                        .nOut(32) //32
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .name("output6")
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        // build multi layer NN
        log.info("Train model...");
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        log.info("\n****************************************** UI SERVER *********************************************\n");
        //Setting up DL4J’s tuning user interface. Use user interface to visualize in the browser
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));



        log.info("\n*************************************** MODEL SUMMARY ******************************************\n");
        log.info(net.summary());

        log.info("\n*************************************** TRAINING **********************************************\n");
        net.fit(trainIter, nEpochs);

        log.info("\n*************************************** EVALUATION **********************************************\n");
        Evaluation trainEva = net.evaluate(trainIter);

        log.info("\n*************************************** TRAINING EVALUATION ******************************************\n");
        log.info(trainEva.stats());

        Evaluation testEva = net.evaluate(testIter);
        log.info("\n*************************************** TESTING EVALUATION ******************************************\n");
        log.info(testEva.stats());

        File modelPath = new File(inputDataDir + "/model.zip"); // save the model for deployment
        ModelSerializer.writeModel(net, modelPath, true);
        log.info("The MINIST model has been saved in {}", modelPath.getPath());

    }
}