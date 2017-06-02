require(mxnet)
data_iris = iris

#data preparation
data_iris[,5] = as.numeric(data_iris[,5])-1
train.ind = c(1:40, 51:85, 115:150)
train.x = data.matrix(data_iris[train.ind, 1:4])
train.y = data_iris[train.ind, 5]
#test = t(data.matrix(data_iris[-train.ind,]))
test.x = data.matrix(data_iris[-train.ind, 1:4])
test.y = data_iris[-train.ind, 5]
devices <- mx.cpu()
mx.set.seed(1234)

#using existing function
model1 <- mx.mlp(train.x, train.y, hidden_node=c(15,15), out_node=3, out_activation="softmax",
                num.round=30, array.batch.size=10, learning.rate=0.07, momentum=0.6, 
                eval.metric=mx.metric.accuracy)

#building new structure of neural network
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=36)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=9)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=3)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

#devices <- mx.cpu()
#mx.set.seed(0)
model2 <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, ctx=devices, num.round=30, 
          array.batch.size=15,learning.rate=0.07, momentum=0.6,  eval.metric=mx.metric.accuracy,
          initializer=mx.init.uniform(0.07), epoch.end.callback=mx.callback.log.train.metric(100))

preds1 <- predict(model1, test.x)
preds2 <- predict(model2, test.x)

preds1.label <- max.col(t(preds1)) - 1
preds2.label <- max.col(t(preds2)) - 1

u =table(test.y,preds1.label)
z =table(test.y,preds2.label)

#Result for test set in model1 and model2
sum(diag(u))/sum(u)
sum(diag(z))/sum(z)

graph.viz(softmax)

