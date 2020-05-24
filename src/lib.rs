pub mod matrix_ops {
    use rand::prelude::*;
    use std::iter;

    pub type Matrix = Vec<Vec<f64>>;

    pub trait MatrixOps {
        fn new_matrix(size: usize) -> Matrix;
        fn init(&mut self);
        fn print(&self);
        fn pad(&self) -> Matrix;
        fn multiply(&self, matrix: &Matrix) -> Matrix;
        fn add(&self, matrix: &Matrix) -> Matrix;
        fn subtract(&self, matrix: &Matrix) -> Matrix;
        fn equals(&self, matrix: &Matrix) -> bool;
        fn get_submatrices(&self) -> [Matrix; 4];
        fn assemble(c1: &Matrix, c2: &Matrix, c3: &Matrix, c4: &Matrix) -> Matrix;
    }

    impl MatrixOps for Matrix {
        fn new_matrix(size: usize) -> Matrix {
            return iter::repeat(iter::repeat(0_f64).take(size).collect()).take(size).collect();
        }
        fn init(&mut self) {
            let mut rng = rand::thread_rng();
            for row in self {
                for element in row.iter_mut() {
                    *element = rng.gen::<f64>();
                }
            }
        }
        fn print(&self) {
            for row in self {
                for element in row {
                    print!("{:width$.3}", element, width = 7);
                }
                println!();
            }
            println!();
        }
        fn pad(&self) -> Matrix {
            let mut pow_of_2 = 1;
            let padding;
            let mut result = self.clone();
            let size = self.len();
            while pow_of_2 < size {
                pow_of_2 *= 2;
            }
            padding = pow_of_2 ;
            for i in 0..padding {
                if i >= result.len() {
                    result.push(Vec::new());
                }
                while result[i].len() < pow_of_2 {
                    result[i].push(0_f64);
                }
            }
            result
        }
        fn multiply(&self, matrix: &Matrix) -> Matrix {
            if self.len() != matrix.len() {
                panic!();
            }
            let size = self.len();
            let mut result = Vec::new();
            for i in 0..size {
                let mut row = Vec::new();
                for j in 0..size {
                    row.push(self[i]
                        .iter()
                        .zip(matrix.iter())
                        .fold(0_f64, |acc, (x, y)| acc + x * y[j])
                    );
                }
                result.push(row);
            }
            result
        }
        fn add(&self, matrix: &Matrix) -> Matrix {
            let size = self.len();
            let mut result = Matrix::new_matrix(size);
            for i in 0..size {
                for j in 0..size {
                    result[i][j] = self[i][j] + matrix[i][j];
                }
            }
            result
        }
        fn subtract(&self, matrix: &Matrix) -> Matrix {
            let size = self.len();
            let mut result = Matrix::new_matrix(size);
            for i in 0..size {
                for j in 0..size {
                    result[i][j] = self[i][j] - matrix[i][j];
                }
            }
            result
        }
        fn equals(&self, matrix: &Matrix) -> bool {
            if self.len() != matrix.len() {
                return false;
            }
            for (row_a, row_b) in self.iter().zip(matrix.iter()) {
                for (a, b) in row_a.iter().zip(row_b.iter()) {
                    if ((a - b) * 1000000.0).round() != 0.0 {
                        return false;
                    }
                }
            }
            return true;
        }
        fn get_submatrices(&self) -> [Matrix; 4] {
            let sub_size = self.len() / 2;
            let mut a = Matrix::new_matrix(sub_size);
            let mut b = Matrix::new_matrix(sub_size);
            let mut c = Matrix::new_matrix(sub_size);
            let mut d = Matrix::new_matrix(sub_size);
            for i in 0..sub_size {
                for j in 0..sub_size {
                    a[i][j] = self[i][j];
                    b[i][j] = self[i][j + sub_size];
                    c[i][j] = self[i + sub_size][j];
                    d[i][j] = self[i + sub_size][j + sub_size];
                }
            }

            [a, b, c, d]
        }
        fn assemble(c1: &Matrix, c2: &Matrix, c3: &Matrix, c4: &Matrix) -> Matrix {
            let sub_size = c1.len();
            let size = sub_size * 2;
            let mut result = Matrix::new_matrix(size);
            for i in 0..sub_size {
                for j in 0..sub_size {
                    result[i][j] = c1[i][j];
                    result[i][j + sub_size] = c2[i][j];
                    result[i + sub_size][j] = c3[i][j];
                    result[i + sub_size][j + sub_size] = c4[i][j];
                }
            }
            result
        }
    }
}