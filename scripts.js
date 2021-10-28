let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');


async function app() {
    console.log('Cargando modelo de identificacion de imagenes');
    net = await mobilenet.load();
    console.log('Carga Terminada');


    //Clasificamos la imagen de carga
    var result = await net.classify(imgEl);
    console.log(result);
    displayImagePrediction();


    //Obtenemos los datos de la webcam
    webcam = await tf.data.webcam(webcamElement);
    while (true) {
        const img = await webcam.capture();
        console.log("datos cam")

        const result = await net.classify(img);

        const activation = net.infer(img, "conv_preds");

        var result2;

        try {
            result2 = await classifier.predicClass(activation);
            const classes = ["Untrained", "Perfume", "Telefono", "Kathy", "Ok", "Rock"]
            document.getElementById('console2:' + classes[result2.label]);
        } catch (error) {
            console.log("Modelo no configurafo aun");
        }


        document.getElementById('console').innerHTML = 'prediction:' + result[0].className +
            " probability:" + result[0].probability;

        img.dispose();

        await tf.nextFrame();


    }
}


imgEl.onload = async function() {
    displayImagePrediction();
}

async function addExample(classId) {
    console.log('added example');
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);

    img.dispose();
}




async function displayImagePrediction() {
    try {
        result = await net.classify(imgEl);
        descEl.innerHTML = JSON.stringify(result);
    } catch (error) {

    }
};

count = 0;

async function cambiarImagen() {
    count = count + 1;
    imgEl.src = "https://picsum.photos/200/300?random=" + count;

}

app();