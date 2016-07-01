TORCH_DIR=$(realpath $(dir $(shell which th))../)

LUA_INCDIR ?= $(TORCH_DIR)/include
LUA_LIBDIR ?= $(TORCH_DIR)/lib

CFLAGS ?=-std=c99 -Wall -pedantic -O2 -I$(LUA_INCDIR)/TH -I$(LUA_INCDIR)

TEST_DIR := tests

all: build

build:
	@echo --- build
	@echo CFLAGS $(CFLAGS)
	@echo LIBFLAG $(LIBFLAG)
	@echo LUA_LIBDIR $(LUA_LIBDIR)
	@echo LUA_BINDIR $(LUA_BINDIR)
	@echo LUA_INCDIR $(LUA_INCDIR)
	@echo LUA $(LUA)

install: 
	@echo --- install
	@echo INST_PREFIX: $(PREFIX)
	@echo INST_BINDIR: $(BINDIR)
	@echo INST_LIBDIR: $(LIBDIR)
	@echo INST_LUADIR: $(LUADIR)
	@echo INST_CONFDIR: $(CONFDIR)

	mkdir -p $(LUADIR)/znn
	cp *.lua $(LUADIR)/znn

test:
	@echo --- test
	find $(TEST_DIR) -name "*.lua" -exec th {} \;

clean:
