import matplotlib.pyplot as plt
import json

f = open('train_test_vals.json')

data = json.load(f)

delete_bleu_history = data['delete']['bleu_history']
delete_retireve_bleu_history = data['delete_retrieve']['bleu_history']
delete_loss_history = data['delete']['loss_history']
delete_retrieve_loss_history = data['delete_retrieve']['loss_history']

plt.plot(delete_bleu_history)
plt.xlabel('Epochs')
plt.ylabel('Bleu Scores')
plt.title('DeleteOnly Model Bleu Scores')
plt.savefig('DeleteOnly_Bleu.png')
plt.show()

plt.plot(delete_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('DeleteOnly Model Training Losses')
plt.savefig('DeleteOnly_Loss.png')
plt.show()

plt.plot(delete_retireve_bleu_history)
plt.xlabel('Epochs')
plt.ylabel('Bleu Scores')
plt.title('DeleteAndRetrieve Model Bleu Scores')
plt.savefig('DeleteAndRetrieve_Bleu.png')
plt.show()

plt.plot(delete_retrieve_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('DeleteAndRetrieve Model Training Losses')
plt.savefig('DeleteAndRetrieve_Loss.png')
plt.show()