def evaluate(args, data, model, target=False):
    model.eval()
    res = [ ]
    m = AUCMeter()
    for idts, idbs, labels in data:
    	if args.layer == 'lstm':
	        model.hidden_1 = model.init_hidden(idts.shape[1])
	        model.hidden_2 = model.init_hidden(idbs.shape[1])

        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        xt = xt.reshape((idts.shape[0], idts.shape[1], args.embed_dim))
        xt = Variable(torch.from_numpy(xt).float())

        xb = embedding_layer.forward(idbs.ravel())
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], args.embed_dim))
        xb = Variable(torch.from_numpy(xb).float())
        
        # build mask
        mt = np.not_equal(idts, padding_id).astype('float')
        mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))
        
        mb = np.not_equal(idbs, padding_id).astype('float')
        mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))

        h_final = model(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())
        h_final = torch.squeeze(h_final)
        
        scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
        scores = torch.squeeze(scores).data.cpu().numpy()
        assert len(scores) == len(labels)
        if target: 
            m.add(scores, labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    if target:
        return m.value(0.05), MAP, MRR, P1, P5
    return MAP, MRR, P1, P5

# helper class used for computing information retrieval metrics, including MAP / MRR / and Precision @ x
class Evaluation():

	def __init__(self, data):

		self.data = data


	def Precision(self, precision_at):
		scores = []
		for item in self.data:
			temp = item[:precision_at]
			if any(val==1 for val in item):
				scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
		return sum(scores)/len(scores) if len(scores) > 0 else 0.0


	def MAP(self):
		scores = []
		missing_MAP = 0
		for item in self.data:
			temp = []
			count = 0.0
			for i,val in enumerate(item):
				if val == 1:
					count += 1.0
					temp.append(count/(i+1))
			if len(temp) > 0:
				scores.append(sum(temp) / len(temp))
			else:
				missing_MAP += 1
		return sum(scores)/len(scores) if len(scores) > 0 else 0.0


	def MRR(self):

		scores = []
		for item in self.data:
			for i,val in enumerate(item):
				if val == 1:
					scores.append(1.0/(i+1))
					break

		return sum(scores)/len(scores) if len(scores) > 0 else 0.0


