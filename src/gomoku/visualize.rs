use image::{ImageBuffer, Rgb, math::Rect};

use crate::{engine::TrainingSample, gomoku::CellState};

use super::{BoardState, GomokuPolicy};

const SQUARE_SIZE: u32 = 20;
const FILL_SIZE: u32 = SQUARE_SIZE * 75 / 100;
const LAST_MOVE_FILL_SIZE: u32 = SQUARE_SIZE * 95 / 100;
const FRAME_SEPARATOR_SIZE: u32 = SQUARE_SIZE / 2;

fn filling_rect(x: u32, y: u32, size: u32) -> Rect {
    let offset = (SQUARE_SIZE - size) / 2;
    Rect {
        x: x + offset,
        y: y + offset,
        width: size,
        height: size,
    }
}

fn latest_move(previous: Option<&BoardState>, current: &BoardState) -> Option<(usize, usize)> {
    let previous = previous?;
    let mut latest = None;
    for x in 0..BoardState::N {
        for y in 0..BoardState::N {
            if previous[(x, y)] == CellState::Empty && current[(x, y)] != CellState::Empty {
                if latest.is_some() {
                    return None;
                }
                latest = Some((x, y));
            }
        }
    }
    latest
}

pub fn generate_game_image(
    history: &[TrainingSample<BoardState, GomokuPolicy>],
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if history.is_empty() {
        return image::RgbImage::new(1, 1);
    }
    let fld = BoardState::N as u32 * SQUARE_SIZE;
    let mut img = image::RgbImage::new(
        fld * history.len() as u32 + FRAME_SEPARATOR_SIZE * (history.len() as u32 - 1),
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

    let mut previous_state = None;
    for (i, sample) in history.iter().enumerate() {
        let mut state = sample.state.clone();
        if i % 2 == 1 {
            state.flip_players_inplace();
        }
        let latest_move = latest_move(previous_state.as_ref(), &state);
        let x = i as u32 * (fld + FRAME_SEPARATOR_SIZE);

        draw_rect(
            &mut img,
            Rect {
                x,
                y: 0,
                width: fld,
                height: fld,
            },
            Rgb([0., 0., 0.]),
        );

        let x_clr = Rgb([255., 0., 0.]);
        let o_clr = Rgb([0., 0., 255.]);

        for i in 0..BoardState::N {
            for j in 0..BoardState::N {
                let clr = match state[(i, j)] {
                    CellState::X => x_clr,
                    CellState::O => o_clr,
                    CellState::Empty => {
                        let probability = sample.policy[(i, j)];
                        if probability == 0.0 {
                            continue;
                        }
                        Rgb([0., probability * 255., 0.])
                    }
                };
                let size = if latest_move == Some((i, j)) {
                    LAST_MOVE_FILL_SIZE
                } else {
                    FILL_SIZE
                };
                draw_rect(
                    &mut img,
                    filling_rect(x + i as u32 * SQUARE_SIZE, j as u32 * SQUARE_SIZE, size),
                    clr,
                );
            }
        }
        previous_state = Some(state);
    }

    img
}

#[cfg(test)]
mod tests {
    use crate::{
        engine::{Game, TrainingSample},
        gomoku::{BoardState, GomokuMove, GomokuPolicy},
    };

    use super::{
        FILL_SIZE, FRAME_SEPARATOR_SIZE, LAST_MOVE_FILL_SIZE, SQUARE_SIZE, generate_game_image,
    };

    fn sample(
        state: BoardState,
        policy_move: GomokuMove,
    ) -> TrainingSample<BoardState, GomokuPolicy> {
        TrainingSample {
            state,
            policy: GomokuPolicy::one_hot(policy_move),
            value: 0.0,
        }
    }

    fn count_color(image: &image::RgbImage, frame: u32, x: u32, y: u32, color: [u8; 3]) -> usize {
        let frame_width = BoardState::N as u32 * SQUARE_SIZE + FRAME_SEPARATOR_SIZE;
        let origin_x = frame * frame_width + x * SQUARE_SIZE;
        let origin_y = y * SQUARE_SIZE;
        (origin_x..origin_x + SQUARE_SIZE)
            .flat_map(|pixel_x| {
                (origin_y..origin_y + SQUARE_SIZE).map(move |pixel_y| (pixel_x, pixel_y))
            })
            .filter(|&(pixel_x, pixel_y)| image.get_pixel(pixel_x, pixel_y).0 == color)
            .count()
    }

    #[test]
    fn draws_policy_and_stones_on_black_fields_and_enlarges_the_latest_move() {
        let first_move = GomokuMove::from_xy(2, 3);
        let second_move = GomokuMove::from_xy(4, 5);
        let third_move = GomokuMove::from_xy(6, 7);
        let initial = BoardState::new();
        let after_first = initial.make_move(&first_move);
        let after_second = after_first.make_move(&second_move);
        let history = [
            sample(initial, first_move),
            sample(after_first, second_move),
            sample(after_second, third_move),
        ];

        let image = generate_game_image(&history);

        assert_eq!(
            count_color(&image, 2, 2, 3, [255, 0, 0]),
            (FILL_SIZE * FILL_SIZE) as usize
        );
        assert_eq!(
            count_color(&image, 2, 4, 5, [0, 0, 255]),
            (LAST_MOVE_FILL_SIZE * LAST_MOVE_FILL_SIZE) as usize
        );
        assert_eq!(
            count_color(&image, 2, 6, 7, [0, 255, 0]),
            (FILL_SIZE * FILL_SIZE) as usize
        );
        assert_eq!(
            count_color(&image, 2, 0, 0, [0, 0, 0]),
            (SQUARE_SIZE * SQUARE_SIZE) as usize
        );
        assert_eq!(
            image.get_pixel(BoardState::N as u32 * SQUARE_SIZE, 0).0,
            [255, 255, 255]
        );
    }
}
