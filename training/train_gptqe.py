import torch
import numpy as np
from models.gpt import GPTQE, GPTConfig
from utils.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, explained_variance_score
from utils.molecule_data import generate_molecule_data
import pennylane as qml
import os

def get_subsequence_energies(op_seq, hamiltonian, init_state, num_qubits):
    """Computes the energies for a sequence of operations."""
    dev = qml.device("default.qubit", wires=num_qubits, shots=1024)

    @qml.qnode(dev)
    def energy_circuit(gqe_ops):
        qml.BasisState(init_state, wires=range(num_qubits))
        for op in gqe_ops:
            qml.apply(op)
        return qml.expval(hamiltonian)

    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(es.item())
    return np.array(energies)

def train_gptqe():
    # Generate molecule data
    molecule_data = generate_molecule_data("H2", use_ucc=True, use_lie_algebra=True)
    h2_data = molecule_data["H2"]
    op_pool = h2_data["op_pool"]
    num_qubits = h2_data["num_qubits"]
    init_state = h2_data["hf_state"]
    hamiltonian = h2_data["hamiltonian"]

    # Training setup
    train_size = 2048
    seq_len = 6
    op_pool_size = len(op_pool)
    train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, seq_len))
    train_op_seq = [[op_pool[idx] for idx in seq] for seq in train_op_pool_inds]
    train_token_seq = np.concatenate([np.zeros(shape=(train_size, 1), dtype=int), train_op_pool_inds + 1], axis=1)

    # Calculate energies for the operator sequences
    train_sub_seq_en = get_subsequence_energies(train_op_seq, hamiltonian, init_state, num_qubits)

    tokens = torch.from_numpy(train_token_seq).to("cuda")
    energies = torch.from_numpy(train_sub_seq_en).to("cuda")

    # Instantiate GPTQE model
    gpt = GPTQE(GPTConfig(
        vocab_size=op_pool_size + 1,
        block_size=seq_len,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=False
    )).to("cuda")

    optimizer = gpt.configure_optimizers(
        weight_decay=0.01,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        device_type="cuda"
    )

    # Training loop
    losses = []
    mae_list = []
    rmse_list = []
    mape_list = []
    explained_var_list = []
    best_mae = float("inf")
    
    os.makedirs(f"./checkpoints/seq_len={seq_len}", exist_ok=True)
    
    for epoch in range(10000):
        gpt.train()  # Ensure the model is in training mode
        epoch_loss = 0
        for token_batch, energy_batch in zip(torch.tensor_split(tokens, 16), torch.tensor_split(energies, 16)):
            optimizer.zero_grad()
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss)

        # Log progress every 500 iterations
        if (epoch + 1) % 100 == 0:
            gpt.eval()  # Set model to evaluation mode
            with torch.no_grad():
                gen_token_seq, pred_Es = gpt.generate(
                    n_sequences=500,
                    max_new_tokens=seq_len,
                    temperature=0.01,
                    device="cuda"
                )
                pred_Es = pred_Es.cpu().numpy()
                gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
                gen_inds = np.clip(gen_inds, 0, op_pool_size - 1)
                gen_op_seq = [[op_pool[idx] for idx in seq] for seq in gen_inds]
                true_Es = get_subsequence_energies(gen_op_seq, hamiltonian, init_state, num_qubits).reshape(-1, 1)

                # Calculate metrics
                mae = mean_absolute_error(true_Es, pred_Es)
                rmse = root_mean_squared_error(true_Es, pred_Es)
                mape = mean_absolute_percentage_error(true_Es, pred_Es)
                explained_var = explained_variance_score(true_Es, pred_Es)

                # Save the metrics for further analysis
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
                explained_var_list.append(explained_var)

                # Print logging information
                print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, Explained Variance: {explained_var:.6f}")

                # Save the model if MAE improves
                if mae < best_mae:
                    best_mae = mae
                    torch.save(gpt.state_dict(), f"./checkpoints/seq_len={seq_len}/gptqe_best_model.pt")
                    print("Saved improved model!")

        # Save the model at regular intervals (e.g., every 1000 epochs)
        if (epoch + 1) % 1000 == 0:
            torch.save(gpt.state_dict(), f"./checkpoints/seq_len={seq_len}/gptqe_epoch_{epoch+1}.pt")
            print(f"Saved model at epoch {epoch + 1}")

    # Save losses and metrics at the end of training
    np.save(f'./checkpoints/seq_len={seq_len}/losses.npy', losses)
    np.save(f'./checkpoints/seq_len={seq_len}/mae_list.npy', mae_list)
    np.save(f'./checkpoints/seq_len={seq_len}/rmse_list.npy', rmse_list)
    np.save(f'./checkpoints/seq_len={seq_len}/mape_list.npy', mape_list)
    np.save(f'./checkpoints/seq_len={seq_len}/explained_var_list.npy', explained_var_list)

if __name__ == "__main__":
    train_gptqe()
