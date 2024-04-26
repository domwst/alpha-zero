use image::{math::Rect, ImageBuffer, Pixel, Rgb};

use crate::tictactoe::CellState;

use super::BoardState;

pub fn generate_game_image(
    history: &[(BoardState, Vec<f32>, f32)],
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let square = 10;
    let fld = 19 * square;
    let line = 5;
    let mut img = image::RgbImage::new(
        fld * history.len() as u32 + line * (history.len() as u32 - 1),
        fld,
    );

    let width = img.width();
    let height = img.height();
    fn draw_rect(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, r: Rect, pixel: Rgb<f32>) {
        for i in r.x..r.x + r.width {
            for j in r.y..r.y + r.height {
                img.put_pixel(i, j, Rgb(pixel.0.map(|v| v as u8)));
            }
        }
    }

    draw_rect(
        &mut img,
        Rect {
            x: 0,
            y: 0,
            width,
            height,
        },
        Rgb([255., 255., 255.]),
    );

    for (i, (state, pol, _)) in history.iter().enumerate() {
        let x = i as u32 * (fld + line);

        let mut pol = pol.iter().copied();
        let x_clr = Rgb([255., 0., 0.]);
        let o_clr = Rgb([0., 0., 255.]);
        let policy = Rgb([0., 255., 0.]);

        for i in 0..19 {
            for j in 0..19 {
                let clr = match state[(i as usize, j as usize)] {
                    CellState::X => x_clr,
                    CellState::O => o_clr,
                    CellState::Empty => {
                        let p = pol.next().unwrap();
                        policy.map(|x| p * x)
                    }
                };
                draw_rect(
                    &mut img,
                    Rect {
                        x: x + i * square,
                        y: 0 + j * square,
                        width: square,
                        height: square,
                    },
                    clr,
                );
            }
        }
    }

    img
}
