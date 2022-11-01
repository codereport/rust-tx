### How to Build

1. Do a `git clone` on this repo
```
git clone git@github.com:codereport/rust-tx.git
```
1. `cd` into repo
```
cd rust-tx
```
2. Make sure you are on `rust nightly`
```
rustup toolchain install nightly
rustup override set nightly
```
3. Run `cargo build` or `cargo test` with `-Z unstable-options`
```
cargo test -Z unstable-options
```

Happy Programming 🙂
