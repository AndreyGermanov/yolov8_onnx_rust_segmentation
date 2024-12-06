use std::sync::Arc;
use image::{GenericImage, GenericImageView, ImageFormat, Rgba};
use image::imageops::FilterType;
use ndarray::{Array, Array2, Axis, IxDyn, s};
use ort::{Environment, SessionBuilder, Value};
use rocket::{response::content,fs::TempFile,form::Form};
use std::path::Path;
#[macro_use] extern crate rocket;

// Main function that defines
// a web service endpoints a starts
// the web service
#[rocket::main]
async fn main() {
    rocket::build()
        .mount("/", routes![index])
        .mount("/detect", routes![detect])
        .launch().await.unwrap();
}

// Site main page handler function.
// Returns Content of index.html file
#[get("/")]
fn index() -> content::RawHtml<String> {
    content::RawHtml(std::fs::read_to_string("index.html").unwrap())
}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes and segmentation masks.
// Returns a JSON array of objects in format [(x1,y1,x2,y2,object_type,probability,mask),..]
#[post("/", data = "<file>")]
fn detect(file: Form<TempFile<'_>>) -> String {
    let buf = std::fs::read(file.path().unwrap_or(Path::new(""))).unwrap_or(vec![]);
    let boxes = detect_objects_on_image(buf,detect_image_format(file));
    return serde_json::to_string(&boxes).unwrap_or_default()
}

fn detect_image_format(file: Form<TempFile<'_>>) -> ImageFormat {
    match file.content_type().unwrap().to_string().split("/").nth(1).unwrap() {
        "png" => ImageFormat::Png,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        _ => ImageFormat::Jpeg
    }
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects,
// their bounding boxes and segmentation masks
// Returns Array of objects in format [(x1,y1,x2,y2,object_type,probability,mask),..]
fn detect_objects_on_image(buf: Vec<u8>, image_format: ImageFormat) -> Vec<(f32,f32,f32,f32,&'static str,f32,Vec<Vec<u8>>)> {
    let (input,img_width,img_height) = prepare_input(buf,image_format);
    let output = run_model(input);
    return process_output(output, img_width, img_height);
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
fn prepare_input(buf: Vec<u8>, image_format: ImageFormat) -> (Array<f32,IxDyn>, u32, u32) {
    let img = image::load_from_memory_with_format(&buf, image_format).unwrap();
    let (img_width, img_height) = (img.width(), img.height());
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r,g,b,_] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    };
    return (input, img_width, img_height);
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw outputs of YOLOv8 network: 1 - detected objects, 2 - segmentation masks
fn run_model(input:Array<f32,IxDyn>) -> (Array<f32,IxDyn>,Array<f32,IxDyn>) {
    let env = Arc::new(Environment::builder().with_name("YOLOv8").build().unwrap());
    let model = SessionBuilder::new(&env).unwrap().with_model_from_file("yolov8m-seg.onnx").unwrap();
    let input_as_values = &input.as_standard_layout();
    let model_inputs = vec![Value::from_array(model.allocator(), input_as_values).unwrap()];
    let outputs = model.run(model_inputs).unwrap();
    let output0 = outputs.get(0).unwrap().try_extract::<f32>().unwrap().view().t().into_owned();
    let output1 = outputs.get(1).unwrap().try_extract::<f32>().unwrap().view().t().into_owned();
    (output0, output1)
}

// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object, the probability and the segmentation mask
// as a 2d array of pixel colors
// Returns array of detected objects in a format [(x1,y1,x2,y2,object_type,probability,mask),..]
fn process_output(outputs:(Array<f32,IxDyn>,Array<f32,IxDyn>),img_width: u32, img_height: u32) -> Vec<(f32,f32,f32,f32,&'static str, f32,Vec<Vec<u8>>)> {
    let (output0, output1) = outputs;
    let boxes_output = output0.slice(s![..,0..84,0]).to_owned();
    let masks_output:Array2<f32> = output1.slice(s![..,..,..,0]).to_owned()
        .into_shape((160*160,32)).unwrap().permuted_axes([1,0]).to_owned();
    let masks_output2:Array2<f32> = output0.slice(s![..,84..116,0]).to_owned();
    let masks = masks_output2.dot(&masks_output).into_shape((8400, 160, 160)).unwrap().to_owned();
    let mut boxes = Vec::new();
    for (index,row) in boxes_output.axis_iter(Axis(0)).enumerate() {
        let row:Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row.iter().skip(4).enumerate()
            .map(|(index,value)| (index,*value))
            .reduce(|accum, row| if row.1>accum.1 { row } else {accum}).unwrap();
        if prob < 0.5 {
            continue
        }
        let mask:Array2<f32>= masks.slice(s![index, .., ..]).to_owned();
        let label = YOLO_CLASSES[class_id];
        let xc = row[0]/640.0*(img_width as f32);
        let yc = row[1]/640.0*(img_height as f32);
        let w = row[2]/640.0*(img_width as f32);
        let h = row[3]/640.0*(img_height as f32);
        let x1 = xc - w/2.0;
        let x2 = xc + w/2.0;
        let y1 = yc - h/2.0;
        let y2 = yc + h/2.0;
        boxes.push((x1,y1,x2,y2,label,prob,process_mask(mask,(x1,y1,x2,y2),img_width,img_height)));
    }

    boxes.sort_by(|box1,box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len()>0 {
        result.push(boxes[0].clone());
        boxes = boxes.iter().filter(|box1| iou(&boxes[0],box1) < 0.7).map(|x| x.clone()).collect()
    }
    return result;
}

// Function transforms the segmentation mask for the object from raw
// 160x160 YOLOv8 output to correct size and returns it as a two dimensional array
fn process_mask(mask:Array2<f32>,rect:(f32,f32,f32,f32),img_width:u32, img_height:u32) -> Vec<Vec<u8>> {
    let (x1,y1,x2,y2) = rect;
    let mut mask_img = image::DynamicImage::new_rgb8(161,161);
    let mut index = 0.0;
    mask.for_each(|item| {
        let color = if *item > 0.0 { Rgba::<u8>([255,255,255,1])  } else { Rgba::<u8>([0,0,0,1]) };
        let y = f32::floor(index / 160.0);
        let x = index - y * 160.0;
        mask_img.put_pixel(x as u32, y as u32, color);
        index += 1.0;
    });
    mask_img = mask_img.crop((x1 / img_width as f32 * 160.0).round() as u32,
                             (y1 / img_height as f32 * 160.0).round() as u32,
                             ((x2-x1) / img_width as f32 * 160.0).round() as u32,
                             ((y2-y1) / img_height as f32 * 160.0).round() as u32
    );
    mask_img = mask_img.resize_exact((x2-x1) as u32,(y2-y1) as u32, FilterType::Nearest);
    let mut result = vec![];
    for y in 0..(y2-y1) as usize {
        let mut row = vec![];
        for x in 0..(x2-x1) as usize {
            let color= mask_img.get_pixel(x as u32, y as u32);
            row.push(*color.0.iter().nth(0).unwrap());
        }
        result.push(row);
    }
    return result;
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
fn iou(box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>), box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>)) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>), box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_,_) = *box2;
    let box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1);
    let box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>), box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_,_) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2-x1)*(y2-y1);
}

// Array of YOLOv8 class labels
const YOLO_CLASSES:[&str;80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];