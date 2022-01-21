package org.deeplearning4j.welding_defect_classification;

import com.sun.scenario.effect.Crop;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.data.ImageWritable;
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
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class ModelDeployment {

    private static Logger log = LoggerFactory.getLogger(ModelDeployment.class);

    public static void main(String[] args) throws Exception{

        int height = 40;
        int width = 40;
        int channels = 1;

        String inputDataDir = "C:/Fan Hao Khong/MKEM Sem 2/MEEM1963 - Deep Learning/Group Project Source/al5083";
        File modelSave = new File(inputDataDir + "/4 class.zip");

        if(!modelSave.exists())
        {
            System.out.println("Model not exist. Abort");
            return;
        }
        File imageToTest = new ClassPathResource("1.png").getFile();

        /*
		#### LAB STEP 1 #####
		Load the saved model
        */
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);

        /*
		#### LAB STEP 2 #####
		Load an image for testing
        */
        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        // Get the image into an INDarray
        INDArray image = loader.asMatrix(imageToTest);


        /*
		#### LAB STEP 3 #####
		[Optional] Preprocessing to 0-1 or 0-255
        */
//        ImageTransform cropImageTransform = new CropImageTransform(450, 150, 100, 150);
//        cropImageTransform.transform(imageToTest);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        /*
		#### LAB STEP 4 #####
		[Optional] Pass to the neural net for prediction
        */

        INDArray output = model.output(image);
        log.info("Label:         " + Nd4j.argMax(output, 1));
        log.info("Probabilities: " + output.toString());
    }

}


