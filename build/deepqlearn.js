Array.prototype.compare = function (array) {
    // if the other array is a falsy value, return
    if (!array)
        return false;

    // compare lengths - can save a lot of time
    if (this.length != array.length)
        return false;

    for (var i = 0, l=this.length; i < l; i++) {
        // Check if we have nested arrays
        if (this[i] instanceof Array && array[i] instanceof Array) {
            // recurse into the nested arrays
            if (!this[i].compare(array[i]))
                return false;
        }
        else if (this[i] != array[i]) {
            // Warning - two different object instances will never be equal: {x:20} != {x:20}
            return false;
        }
    }
    return true;
}

function DeepLearning(gameManager) {
    this.gameManager = gameManager;
    this.lastScore = this.gameManager.score;
    this.lastGameGrid = this.convertGrid(this.gameManager.grid);
    
    var num_inputs = 16;
    var num_actions = 5;
    var temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
    var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

    // the value function network computes a value of taking any of the possible actions
    // given an input state. Here we specify one explicitly the hard way
    // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
    // to just insert simple relu hidden layers.
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'regression', num_neurons:num_actions});

    // options for the Temporal Difference learner that trains the above net
    // by backpropping the temporal difference learning rule.
    var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

    var opt = {};
    opt.temporal_window = temporal_window;
    opt.experience_size = 30000;
    opt.start_learn_threshold = 1000;
    opt.gamma = 0.7;
    opt.learning_steps_total = 200000;
    opt.learning_steps_burnin = 3000;
    opt.epsilon_min = 0.05;
    opt.epsilon_test_time = 0.05;
    opt.layer_defs = layer_defs;
    opt.tdtrainer_options = tdtrainer_options;

    this.brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo
    var lastLearningState = this.gameManager.storageManager.getAIState();
    if (lastLearningState) {
        this.brain.value_net.fromJSON(lastLearningState);
    }
    this.reward_graph = new cnnvis.Graph();
    

    document.getElementsByClassName('restart-button')[0].onclick = this.start();
}

DeepLearning.prototype = {
    simulateKeyPress: function (keyCode) {
//      var keyboardEvent = document.createEvent("KeyboardEvent");
//      var initMethod = typeof keyboardEvent.initKeyboardEvent !== 'undefined' ? "initKeyboardEvent" : "initKeyEvent";
//
//      keyboardEvent[initMethod](
//                          "keypress", // event type : keydown, keyup, keypress, "keydown"
//                          true, // bubbles
//                          true, // cancelable
//                          window, // viewArg: should be window
//                          false, // ctrlKeyArg
//                          false, // altKeyArg
//                          false, // shiftKeyArg
//                          false, // metaKeyArg
//                          keyCode, // keyCodeArg : unsigned long the virtual key code, else 0
//                          0 // charCodeArgs : unsigned long the Unicode character associated with the depressed key, else 0
//      );
//      document.dispatchEvent(keyboardEvent);

        var eventObj = document.createEventObject ?
            document.createEventObject() : document.createEvent("Events");

        if(eventObj.initEvent){
          eventObj.initEvent("keydown", true, true);
        }

        eventObj.keyCode = keyCode;
        eventObj.which = keyCode;

        document.dispatchEvent ? document.dispatchEvent(eventObj) : document.fireEvent("onkeydown", eventObj);
    },

    convertGrid: function (grid) {
        var result = {
            newGrid: [],
            max: 0,
            emptyCount: 0
        }
        for (i=0; i<grid.size; i++) {
            for(j=0; j<grid.size; j++) {
                var location = i*4 + j;
                if (grid.cells[i][j]) {
                    result.newGrid[location] = grid.cells[i][j].value;
                    result.max = result.max < grid.cells[i][j].value ? grid.cells[i][j].value : result.max;
                }
                else {
                    result.newGrid[location] = 0;
                    result.emptyCount++;
                }

            }
        }
        return result;
    },

    makeAction: function () {
        this.currentGameGrid = this.convertGrid(this.gameManager.grid);
        var actionId = this.brain.forward(this.currentGameGrid.newGrid);
        if(actionId > 3) {
            console.warn('action: ' + actionId);
        }

        switch (actionId) {
            case 0:
                this.simulateKeyPress(37);
                break;
            case 1:
                this.simulateKeyPress(38);
                break;
            case 2:
                this.simulateKeyPress(39);
                break;
            case 3:
                this.simulateKeyPress(40);
                break;
            case 4:
                document.getElementsByClassName('restart-button')[0].click();
                break;
                
        }
        this.backward(this.currentGameGrid);
        this.draw_stats();
        this.draw_net();
        this.brain.visSelf(document.getElementById('brain_info_div'));
    },

    backward: function (lastGameGrid) {
        var score = this.gameManager.score;
        var reward = 0;
        var maxScore = this.gameManager.storageManager.getBestScore();
        this.currentGameGrid = this.convertGrid(this.gameManager.grid);

        if (score > 0) {
            reward += Math.log(score) / Math.log(2);
        }

        if (score >= maxScore) {
            reward = reward + 1;
        }

        if (lastGameGrid.max < this.currentGameGrid.max) {
            var delta = (this.currentGameGrid.max - lastGameGrid.max);
            reward += Math.log(delta) / Math.log(2);
        }

        if (lastGameGrid.emptyCount !== 0)

//        var scoreDelta = this.gameManager.score - this.lastScore;
//        if (scoreDelta <= 0 && score === 0 && lastGameGrid.emptyCount > 0) {
//            reward = reward - 4;
//        }
//
//        if (scoreDelta < 0 && score === 0) {
//            scoreDelta = -1; // new Game
//        }
//        else {
//            scoreDelta = 2 + scoreDelta / 10;
//        }
//
//        reward = reward + scoreDelta;
//
//        this.lastScore = this.gameManager.score;
//
//        // no change was done? bad reward
//        if (lastGameGrid.newGrid.compare(this.currentGameGrid.newGrid)) {
//            //console.log('reward: 0');
//            //return this.brain.backward(0);
//            reward = reward - 2;
//        }

        // awarding for more empty cells
        //if (this.currentGameGrid.emptyCount > lastGameGrid.emptyCount) {
        //    reward += (16 / (lastGameGrid.emptyCount - this.currentGameGrid.emptyCount));
        //}

        //reward += scoreDelta; // / 1000;

        if (this.gameManager.won) {
            reward += 2048;
        }

//        if (this.gameManager.over) {
//            var _this = this;
//            setTimeout(function() {
//                _this.gameManager.storageManager.setAIState(_this.brain.value_net.toJSON());
//                document.getElementsByClassName('retry-button')[0].click();
//            }, 5);
//            //reward = -10;
//        }

        console.log('reward: ' + reward);
        this.brain.backward(reward);
    },

    start: function() {
        this.interval = setInterval(this.makeAction.bind(this), 200);
    },
    
    draw_net: function() {
      var canvas = document.getElementById("net_canvas");
      var ctx = canvas.getContext("2d");
      var W = canvas.width;
      var H = canvas.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var L = this.brain.value_net.layers;
      var dx = (W - 50)/L.length;
      var x = 10;
      var y = 40;
      ctx.font="12px Verdana";
      ctx.fillStyle = "rgb(0,0,0)";
      ctx.fillText("Value Function Approximating Neural Network:", 10, 14);
      for(var k=0;k<L.length;k++) {
        if(typeof(L[k].out_act)==='undefined') continue; // maybe not yet ready
        var kw = L[k].out_act.w;
        var n = kw.length;
        var dy = (H-50)/n;
        ctx.fillStyle = "rgb(0,0,0)";
        ctx.fillText(L[k].layer_type + "(" + n + ")", x, 35);
        for(var q=0;q<n;q++) {
          var v = Math.floor(kw[q]*100);
          if(v >= 0) ctx.fillStyle = "rgb(0,0," + v + ")";
          if(v < 0) ctx.fillStyle = "rgb(" + (-v) + ",0,0)";
          ctx.fillRect(x,y,10,10);
          y += 12;
          if(y>H-25) { y = 40; x += 12};
        }
        x += 50;
        y = 40;
      }
    },
    
    draw_stats: function() {
      var canvas = document.getElementById("vis_canvas");
      var ctx = canvas.getContext("2d");
      var W = canvas.width;
      var H = canvas.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var b = this.brain;
      var netin = b.last_input_array;
      ctx.strokeStyle = "rgb(0,0,0)";
      //ctx.font="12px Verdana";
      //ctx.fillText("Current state:",10,10);
      ctx.lineWidth = 10;
      ctx.beginPath();
      for(var k=0,n=netin.length;k<n;k++) {
        ctx.moveTo(10+k*12, 120);
        ctx.lineTo(10+k*12, 120 - netin[k] * 100);
      }
      ctx.stroke();
      
        this.reward_graph.add(1, b.average_reward_window.get_average());
        var gcanvas = document.getElementById("graph_canvas");
        this.reward_graph.drawSelf(gcanvas);
    }


}
