self.GLOBAL = {
  isWindow: function() { return false; },
  isWorker: function() { return true; },
};
importScripts("../resources/testharness.js");
importScripts("../resources/WebIDLParser.js")
importScripts("../resources/idlharness.js")
importScripts("./dist/webnn-polyfill.js")
importScripts("./idlharness.https.any.js");
done();