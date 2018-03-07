#!/bin/sh
cargo test &&
cargo build --release &&
cp -f target/release/paneb.dll ../paneb-unity/Assets/paneb.dll
