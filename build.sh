#!/usr/bin/env bash

#abort if any command fails
set -e

mkdir -p build/autoconf
builddir=$(readlink -f build)
debug="--enable-debug"

export CPPFLAGS=-I$builddir/include
export CFLAGS=-I$builddir/include
export LDFLAGS=-L$builddir/lib

build () {
    echo building $1
    path=$1
    url=$2
    branch=$3
    if [ ! -d $path ]; then
        git clone $url $path;
    else
        pushd $path; git pull origin $branch; popd
    fi
    pushd $1
        mkdir -p build/autoconf
        autoreconf -i
        ./configure --prefix=$builddir $debug
        make
        make install
    popd
    echo
}

echo
echo builddir = $builddir
echo

build libaesrand    https://github.com/5GenCrypto/libaesrand master

autoreconf -i
./configure $debug
make
