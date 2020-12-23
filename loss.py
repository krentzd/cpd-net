import torch
# import sugartensor as tf

def chamfer_loss(A, B):
    r_A = torch.sum(A * A, dim=2)
    r_A = torch.reshape(r_A, [int(r_A.shape[0]), int(r_A.shape[1]), 1])

    r_B = torch.sum(B * B, dim=2)
    r_B = torch.reshape(r_B, [int(r_A.shape[0]), int(r_A.shape[1]), 1])

    t = (r_A - 2 * torch.matmul(A, torch.transpose(B, 2, 1)) + torch.transpose(r_B, 2, 1))

    return torch.mean((torch.min(t, dim=1)[0] + torch.min(t, dim=2)[0])[0] / 2.0)

# def chamfer_loss_tf(A,B):
#     r=tf.reduce_sum(A*A,2)
#     r=tf.reshape(r,[int(r.shape[0]),int(r.shape[1]),1])
#     r2=tf.reduce_sum(B*B,2)
#     r2=tf.reshape(r2,[int(r.shape[0]),int(r.shape[1]),1])
#
#     t=(r-2*tf.matmul(A, tf.transpose(B,perm=[0, 2, 1])) + tf.transpose(r2,perm=[0, 2, 1]))
#
#     loss = tf.reduce_mean((tf.reduce_min(t, axis=1)+tf.reduce_min(t,axis=2))/2.0)
#
#     return loss
#
# if __name__ == '__main__':
#
#     X = torch.randn(100, 3).unsqueeze(0)
#     Y = torch.randn(100, 3).unsqueeze(0)
#     print(X)
#
#     with tf.Session() as sess:
#         X_tf = tf.convert_to_tensor(X.numpy())
#         Y_tf = tf.convert_to_tensor(Y.numpy())
#         print(X_tf.eval(session=sess))
#         print('Torch ', chamfer_loss(X, Y))
#         print('TF ', chamfer_loss_tf(X_tf, Y_tf).eval(session=sess))
