extern crate nalgebra as na;
extern crate png;

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::path::Path;
use std::io::BufWriter;
// To use encoder.set()
use png::HasParameters;

fn main() {
    let f = File::open("cifar-10-batches-bin/data_batch_1.bin").unwrap();
    let mut reader = BufReader::new(f);

    let cfar_m = 10000usize;
    let cfar_n = 3072usize;
    let cfar_img_byte_stride = 3073usize;
    let mut X: na::DMatrix<u8> = na::DMatrix::identity(cfar_n, cfar_m);
    let mut Y: na::DVector<u8> = na::DVector::identity(cfar_m);

    let multiplier = cfar_img_byte_stride;
    for i in 0..cfar_m {
        // Read first byte which tells us the category of the image
        let mut label = [0u8];
        let read = reader.read(&mut label).unwrap();
        assert_eq!(read, 1);

        // If cat, set it as one
        if label[0] == 3 {
            Y[i] = 1;
            println!("We are at i = {}", i);
        }
        // If not, set it as zero
        else {
            Y[i] = 0;
        }

        // Read the data of an image into a column of the data vector
        let mut img_data = vec![0; cfar_n];
        reader.read_exact(&mut img_data).unwrap();

        if label[0] == 3 {
            let path = Path::new(r"image.png");
            let file = File::create(path).unwrap();
            let ref mut w = BufWriter::new(file);

            let mut encoder = png::Encoder::new(w, 32, 32); // Width is 2 pixels and height is 1.
            encoder.set(png::ColorType::RGB).set(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();

            let mut data = vec![0u8; cfar_n];
            for n in 0..1024 {
                let nn = n * 3;
                data[nn] = img_data[n];
                data[nn + 1] = img_data[n + 1024];
                data[nn + 2] = img_data[n + 2048];
            }

            writer.write_image_data(&data).unwrap(); // Save
        }

    }

    //println!("Y content is: {}", Y);

}
