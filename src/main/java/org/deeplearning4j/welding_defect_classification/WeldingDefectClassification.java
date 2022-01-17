package org.deeplearning4j.welding_defect_classification;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
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
        int n = 8;
        int height = 600/n;  // 输入图像高度
        int width = 800/n;   // 输入图像宽度
        int channels = 1; // 输入图像通道数
        int outputNum = 3; // 3分类
        int batchSize = 64;
        int nEpochs = 1;
        int seed = 1234;
        Random randNumGen = new Random(seed);
        String inputDataDir = "D:/al5083";

// 训练数据的向量化


        File data = new File(inputDataDir + "/test");
        FileSplit fileSplit = new FileSplit(data);
        //create random path filter using RandomPathFilter
        RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, NativeImageLoader.ALLOWED_FORMATS);

        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 70, 30);
        InputSplit trainSplit = filesInDirSplit[0];
        InputSplit testSplit = filesInDirSplit[1];

//        File trainData = new File(inputDataDir + "/train");
//        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        ImageTransform cropImageTransform = new CropImageTransform(320, 0, 0, 0);
        trainRR.initialize(trainSplit, cropImageTransform);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

// 将像素从0-255缩放到0-1 (用min-max的方式进行缩放)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

// 测试数据的向量化
//        File testData = new File(inputDataDir + "/test");
//        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit, cropImageTransform);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(scaler); // same normalization for better results

// 设置网络层及超参数

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(5e-5))
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .name("Conv1")
                        .nIn(channels)
                        .nOut(16)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .padding(2,2)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .name("Pooling1")
                        .poolingType(PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .name("Conv2")
                        .nOut(16)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .padding(2,2)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .name("Pooling2")
                        .poolingType(PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .build())
//                .layer(4,new ConvolutionLayer.Builder()
//                        .name("Conv3")
//                        .nOut(16)
//                        .kernelSize(3,3)
//                        .stride(1,1)
//                        .padding(2,2)
//                        .build())
//                .layer(5,new SubsamplingLayer.Builder()
//                        .name("Pooling3")
//                        .poolingType(PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(1,1)
//                        .build())
                .layer(4,new DenseLayer.Builder()
                        .name("dense1")
                        .nOut(64)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(5,new OutputLayer.Builder()
                        .name("output")
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        // 新建一个多层网络模型
        log.info("Train model...");
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        log.info("\n****************************************** UI SERVER *********************************************\n");
        //Setting up DL4J’s tuning user interface
        //Use user interface to visualize in the browser
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

    }
}