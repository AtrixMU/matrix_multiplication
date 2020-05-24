use std::env;
use std::time::Instant;

use matrix_multiplication::matrix_ops::{Matrix, MatrixOps};


fn strassen(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
    let size = matrix_a.len();
    if size != matrix_b.len() {
        panic!();
    }
    let a = matrix_a.pad();
    let b = matrix_b.pad();

    let mut result = strassen_mult(&a, &b);
    while result.len() > size {
        result.pop();
    }
    for row in result.iter_mut() {
        while row.len() > size {
            row.pop();
        }
    }
    result
}

fn strassen_mult(a: &Matrix, b: &Matrix) -> Matrix {
    let [a1, a2, a3, a4] = a.get_submatrices();
    let [b1, b2, b3, b4] = b.get_submatrices();

    if a1.len() == 1 {
        let (a1, a2, a3, a4) = (a1[0][0], a2[0][0], a3[0][0], a4[0][0]);
        let (b1, b2, b3, b4) = (b1[0][0], b2[0][0], b3[0][0], b4[0][0]);
        let m1 = (a1 + a4) * (b1 + b4);
        let m2 = (a3 + a4) * b1;
        let m3 = a1 * (b2 - b4);
        let m4 = a4 * (b3 - b1);
        let m5 = (a1 + a2) * b4;
        let m6 = (a3 - a1) * (b1 + b2);
        let m7 = (a2 - a4) * (b3 + b4);

        let mut result = Matrix::new_matrix(2);
        result[0][0] = m1 + m4 - m5 + m7;
        result[0][1] = m3 + m5;
        result[1][0] = m2 + m4;
        result[1][1] = m1 - m2 + m3 + m6;
        return result;
    }

    let m1 = strassen_mult(&a1.add(&a4), &b1.add(&b4));
    let m2 = strassen_mult(&a3.add(&a4), &b1);
    let m3 = strassen_mult(&a1, &b2.subtract(&b4));
    let m4 = strassen_mult(&a4, &b3.subtract(&b1));
    let m5 = strassen_mult(&a1.add(&a2), &b4);
    let m6 = strassen_mult(&a3.subtract(&a1), &b1.add(&b2));
    let m7 = strassen_mult(&a2.subtract(&a4), &b3.add(&b4));

    let c1 = m1.add(&m4).subtract(&m5).add(&m7);
    let c2 = m3.add(&m5);
    let c3 = m2.add(&m4);
    let c4 = m1.subtract(&m2).add(&m3).add(&m6);

    return Matrix::assemble(&c1, &c2, &c3, &c4);
}


fn main() {
    let size;
    let mode;
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Error: invalid arguments.");
        println!("The first argument selects the multiplication method. 1 - standard, 2 - Strassen.");
        println!("The second argument selects the order of matrices.");
        println!("Example: program_name.exe 1 256");
        return;
    }
    mode = args[1].parse::<usize>().unwrap();
    size = args[2].parse::<usize>().unwrap();
    let mut a = Matrix::new_matrix(size);
    let mut b = Matrix::new_matrix(size);
    a.init();
    b.init();

    match mode {
        1 => {
            let now = Instant::now();
            let _res = a.multiply(&b);
            let result = now.elapsed().as_secs_f64();
            println!("Size: {}\nTime: {}", size, result);
        },
        2 => {
            let now = Instant::now();
            let _res = strassen(&a, &b);
            let result = now.elapsed().as_secs_f64();
            println!("Size: {}\nTime: {}", size, result);
        },
        _ => {
            println!("Invalid mode specified.");
            return;
        }
    }
}
