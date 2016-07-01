package = "znn"
version = "scm-1"
source = {
   url = "https://github.com/iamalbert/zd.git",
   tag = "master"
}
description = {
   summary = "my extension to torch7, totaly experimental and unstable",
   detailed = [[
       my extension to torch7, totaly experimental and unstable
   ]],
   homepage = "https://github.com/iamalbert/znn",
   license = "BSD"
}
dependencies = {
   "torch >= 7.0",
   "nn",
}
build = {
   type = "make",
   build_variables = {
      CFLAGS = "-std=c99 -Wall -pedantic -O2 -I$(LUA_INCDIR)/TH -I$(LUA_INCDIR)",
      LIBFLAG = "$(LIBFLAG)",
      LUA = "$(LUA)",
      LUA_BINDIR = "$(LUA_BINDIR)",
      LUA_INCDIR = "$(LUA_INCDIR)",
      LUA_LIBDIR = "$(LUA_LIBDIR)"
   },
   install_variables = {
      BINDIR = "$(BINDIR)",
      CONFDIR = "$(CONFDIR)",
      LIBDIR = "$(LIBDIR)",
      LUADIR = "$(LUADIR)",
      PREFIX = "$(PREFIX)"
   }
}
