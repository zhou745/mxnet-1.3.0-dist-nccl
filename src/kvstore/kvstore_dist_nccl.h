/*
This module is write to implement a distributed version of
nccl kvstore.
*/

#ifndef MXNET_KVSTORE_KVSTORE_DIST_NCCL_H_
#define MXNET_KVSTORE_KVSTORE_DIST_NCCL_H_

//include essential head files for nccl
#if MXNET_USE_NCCL

#include <mxnet/kvstore.h>
#include <nccl.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include <tuple>
#include "./comm.h"
#include "./kvstore_local.h"
#include "../common/cuda_utils.h"

// NCCL v2 introduces NCCL_MAJOR macro for versioning,
// so if there is no such macro defined in nccl.h
// then it is NCCL v1
#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#endif

#if NCCL_MAJOR == 1
#define ncclGroupStart()
#define ncclGroupEnd()
#define ncclNumTypes nccl_NUM_TYPES
#endif  // NCCL_MAJOR == 1

//include essential files for dist kvstore
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif

#include<iostream>

namespace mxnet{
namespace kvstore{
// dist nccl kvstore
class KVStoreDistNccl : public KVStoreLocal{
  public:

    explicit KVStoreDistNccl() : KVStoreLocal(), ps_worker_(nullptr), server_(nullptr){
        if(IsWorkerNode()){
            ps_worker_ = new ps::KVWorker<real_t>(0);
            ps::StartAsync("mxnet\0");
            if (!ps::Postoffice::Get()->is_recovery()) {
               ps::Postoffice::Get()->Barrier(
               ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
            }
         }
         bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);

         pinned_ctx_ = Context::CPUPinned(0);
         nccl_init_=false;
     }

    virtual ~KVStoreDistNccl() {
         Engine::Get()->WaitForAll();
         if (IsWorkerNode()) {
            if (barrier_before_exit_) {
                Barrier();
                if (get_rank() == 0) {
                   // stop the executor at servers
                   SendCommandToServers(static_cast<int>(CommandType::kStopServer), "");
                 }
            }
            ps::Finalize(barrier_before_exit_);
            delete ps_worker_;
            for (auto e : nccl_data_) {
               cudaStreamDestroy(e.second.stream);
               ncclCommDestroy(e.second.comm);
             }
         }
    }

    //directly copied from kvstore_dist

    void set_updater(const Updater& updater) override {
         CHECK(updater) << "invalid updater";
         if (IsServerNode()) {
            CHECK_NOTNULL(server_)->set_updater(updater);
         } else {
            updater_ = updater;
         }
    }

    void Barrier() override {
         ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    }

    void SendCommandToServers(int cmd_id, const std::string& cmd_body) override {
         CHECK_NOTNULL(ps_worker_);
         ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
    }

    int get_group_size() const override { return ps::NumWorkers(); }

    int get_rank() const override { return ps::MyRank(); }

    int get_num_dead_node(int node_id, int timeout) const override {
        int number = 0;
        auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
        const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
        std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
        for (int r : dead_nodes) {
            if (watch_set.find(r) != watch_set.end()) number++;
        }
        return number;
     }

    void RunServer(const Controller& controller) override {
         CHECK(!IsWorkerNode());
         if (IsServerNode()) {
            server_ = new KVStoreDistServer();
            server_->set_controller(controller);
         }

         ps::StartAsync("mxnet_server\0");
         if (!ps::Postoffice::Get()->is_recovery()) {
             ps::Postoffice::Get()->Barrier(
             ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
         }
         if (server_) server_->Run();
         ps::Finalize();
         if (server_) {
             delete server_;
         }
         server_ = nullptr;
    }


private:

     struct PSKV {
            ps::SArray<ps::Key> keys;
            ps::SArray<int> lens;
            int size;
     };

     std::unordered_map<int, PSKV> ps_kv_;

     std::mutex mu_;

     void InitImpl(const std::vector<int>& keys,const std::vector<NDArray>& values) override {
          //copy the initial data to remote server
          CheckUnique(keys);
          if (get_rank() == 0) {
               Push_(keys, values, 0, false);
               // wait until the push is finished
               for (const int key : keys) {
               comm_buf_[key].WaitToWrite();
               }
          } else {
                    // do nothing
          }
          if (!ps::Postoffice::Get()->is_recovery()) {
             Barrier();
          }
     }

     //wraper initialization need no merge
     void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override{
                   Push_(keys,values,priority,true);
                }

     void Push_(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority,
                bool domerge) {
          std::vector<int> uniq_keys;
          std::vector<std::vector<NDArray> > grouped_vals;
          GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);
          // merge over devices using nccl
          for (size_t i = 0; i < uniq_keys.size(); ++i) {
              int key = uniq_keys[i];
              const auto& vals = grouped_vals[i];

              auto& merge_buf_devs = merge_buf_devs_[key];
              auto& merge_buf = merge_buf_[key];

              //check if nccl initialized, note that we do not wish it to initialize in kv init
              if(!nccl_init_ && domerge){
                  for(const auto& d : vals){
                      devs_.push_back(d.ctx());
                  }
                  InitNCCL(devs_);
                  nccl_init_=true;
              }

              //check if the two merging buffs have already been initialized
              if(merge_buf_devs.is_none() && domerge){
                  merge_buf_devs = NDArray(vals[0].shape(),devs_[0],false,vals[0].dtype());
              }
              if(merge_buf.is_none()){
                  merge_buf=NDArray(vals[0].shape(),pinned_ctx_,false,vals[0].dtype());
              }

              if(domerge) {
                 //reduce the gradient on an graphic card buffer
                 Reduce(key,vals,merge_buf_devs,priority);
                 //copy the buffer to memory
                 CopyFromTo(merge_buf_devs,merge_buf,priority);
              } else {
                 //copy one copy of the data bto the buffer
                 CopyFromTo(vals[0],merge_buf,priority);
              }
              auto &comm_buf = comm_buf_[key];
              //assign merged to comm_buf
              comm_buf = merge_buf;
              //now start to push
              PSKV& pskv = EncodeDefaultKey(key, comm_buf.shape().Size(), true);

              PushDefault(key, comm_buf, pskv, priority);
          }
     }

     void PushDefault(int key, const NDArray &send_buf, const PSKV& pskv, int priority) {
         auto push_to_servers =
         [this, key, pskv, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
             // convert to ps keys
             size_t size = send_buf.shape().Size();
             real_t* data = send_buf.data().dptr<real_t>();
#if MKL_EXPERIMENTAL == 1
          mkl_set_tblob_eager_mode(send_buf.data());
#endif
             // do push. false means no delete
             ps::SArray<real_t> vals(data, size, false);
             CHECK_NOTNULL(ps_worker_)->ZPush(
                pskv.keys, vals, pskv.lens,
                static_cast<int>(DataHandleType::kDefaultPushPull), [cb]() { cb(); });
             };
             Engine::Get()->PushAsync(
                 push_to_servers,
                 pinned_ctx_,
                 {send_buf.var()},
                 {},
                 FnProperty::kNormal,
                 priority,
                 PROFILER_MESSAGE("KVStoreDistDefaultPush"));
             }

      void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
           std::vector<int> uniq_keys;
           std::vector<std::vector<NDArray*> > grouped_vals;
           GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

           for(size_t i=0;i<uniq_keys.size();++i){
              int key = uniq_keys[i];
              //note that we are using the same buffer as push does to ensure order
              auto& recv_buf = comm_buf_[key];
              //check if comm_buf_ initialized
              if (recv_buf.is_none()) {
                   recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                                             true, grouped_vals[i][0]->dtype());
              }
              //lambda for pull from remote server
              auto pull_from_servers = [this, key, recv_buf](
                   RunContext rctx, Engine::CallbackOnComplete cb) {
                   size_t size = recv_buf.shape().Size();

                   PSKV& pskv = EncodeDefaultKey(key, size, false);
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(recv_buf.data());
#endif
                   real_t* data = recv_buf.data().dptr<real_t>();
                   // false means not to delete data when SArray is deleted
                   auto vals = new ps::SArray<real_t>(data, size, false);
                   // issue pull
                  int cmd = static_cast<int>(DataHandleType::kDefaultPushPull);

                  CHECK_NOTNULL(ps_worker_)->ZPull(
                       pskv.keys, vals, &pskv.lens, cmd, [vals, cb](){ delete vals; cb(); });
                  };

                  CHECK_NOTNULL(Engine::Get())->PushAsync(
                       pull_from_servers,
                       pinned_ctx_,
                       {},
                       {recv_buf.var()},
                       FnProperty::kNormal,
                       priority,
                       PROFILER_MESSAGE("KVStoreDistDefaultStoragePull"));
                  if(nccl_init_){
                    Broadcast(key,recv_buf,grouped_vals[i],priority); //nccl broad cast if the first time is done
                  } else {
                    for(size_t j = 0; j < grouped_vals[i].size(); ++j){
                       CopyFromTo(recv_buf, *(grouped_vals[i][j]), priority);
                    }
                  }
           }
      }

     //nccl reduce
     virtual void Reduce(const int key,
                 const std::vector<NDArray>& src,
                 NDArray& merge_buf_devs,
                 int priority) {
                 std::vector<Engine::VarHandle> const_vars;
                 //set the dependency between backward and reduce
                 for (size_t id = 0; id < src.size(); ++id) {
                        const_vars.push_back(src[id].var());
                 }
                 //lambda for reduce
                 auto merge_on_gpu = [this, src, merge_buf_devs](RunContext rctx,Engine::CallbackOnComplete cb){
                   std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
                   int root = merge_buf_devs.ctx().dev_id;
                   ncclGroupStart();
                   for(size_t i=0; i<src.size(); ++i){
                      NCCLEntry cur = nccl_data_[src[i].ctx().dev_id];
                      MSHADOW_TYPE_SWITCH(src[i].dtype(),Dtype,
                         ncclReduce(src[i].data().dptr<Dtype>(),
                                    merge_buf_devs.data().dptr<Dtype>(),
                                    src[i].shape().Size(),
                                    GetNCCLType(src[i].dtype()),
                                    ncclSum,
                                    root,
                                    cur.comm,
                                    cur.stream););
                   }
                   ncclGroupEnd();
                   //wait for all the op to be completed
                   for (auto cur : nccl_data_) {
                      CUDA_CALL(cudaSetDevice(cur.second.dev_id));
                      CUDA_CALL(cudaStreamSynchronize(cur.second.stream));
                   }
                   cb();
                 };
                 //push to engine for reduce
                 Engine::Get()->PushAsync(merge_on_gpu,
                                          Context::CPU(),
                                          const_vars,
                                          {merge_buf_devs.var()},
                                          FnProperty::kCPUPrioritized,
                                          priority,
                                          PROFILER_MESSAGE("KVStoreReduce"));
                  //Engine::Get()->WaitForAll();
                 }

     virtual void Broadcast(const int key,
                            const NDArray& src,
                            const std::vector<NDArray*>& dst,
                            int priority) {
           auto& buf_brd = merge_buf_devs_[key];
           std::vector<Engine::VarHandle> mutable_vars;
           //set the dependency between forward and broadcast
           for(size_t id=0;id<dst.size();++id){
              mutable_vars.push_back(dst[id]->var());
           }
           //copy the value in buffer to the root graphic card memory first
           CopyFromTo(src,buf_brd,priority);

           auto broad_cast_ = [buf_brd, dst, this](RunContext rctx,Engine::CallbackOnComplete cb){
                std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
                int root = buf_brd.ctx().dev_id;
                ncclGroupStart();
                for(size_t i = 0; i < dst.size(); ++i){
                   auto& bcast = *dst[i];
                   NCCLEntry cur = nccl_data_[bcast.ctx().dev_id];
                   MSHADOW_TYPE_SWITCH(bcast.dtype(), DType,
                                       ncclBroadcast(buf_brd.data().dptr<DType>(),
                                                 bcast.data().dptr<DType>(),
                                                 bcast.shape().Size(),
                                                 GetNCCLType(bcast.dtype()),
                                                 root,
                                                 cur.comm,
                                                 cur.stream););
                }
                ncclGroupEnd();
                //wait for all the op to be completed
                for (auto cur : nccl_data_) {
                    CUDA_CALL(cudaSetDevice(cur.second.dev_id));
                    CUDA_CALL(cudaStreamSynchronize(cur.second.stream));
                }
                cb();
                //Engine::Get()->WaitForAll();
           };

           Engine::Get()->PushAsync(
                          broad_cast_,
                          Context::CPU(),
                          {buf_brd.var()},
                          mutable_vars,
                          FnProperty::kCPUPrioritized,
                          priority,
                          PROFILER_MESSAGE("KVStoreBCast"));
     }


     inline PSKV& EncodeDefaultKey(int key, size_t size, bool is_push) {
         mu_.lock();
         PSKV& pskv = ps_kv_[key];
         mu_.unlock();
         if (!pskv.keys.empty()) {
             CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
         } else {
             auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
             int num_servers = krs.size();
             CHECK_GT(num_servers, 0);

             // a simple heuristic for load balance
             if (size < bigarray_bound_) {
             // send it to a single random picked server
                 int server = (key * 9973) % num_servers;
                 ps::Key ps_key = krs[server].begin() + key;
                 CHECK_LT(ps_key, krs[server].end());
                 pskv.keys.push_back(ps_key);
                 pskv.lens.push_back(size);
                 pskv.size = size;
             } else {
             // parition it to all servers
             pskv.size = 0;
             for (int i = 0; i < num_servers; ++i) {
                 size_t part_size =
                 static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
                 static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));
                 ps::Key ps_key = krs[i].begin() + key;
                 CHECK_LT(ps_key, krs[i].end());
                 pskv.keys.push_back(ps_key);
                 pskv.lens.push_back(part_size);
                 pskv.size += part_size;
             }
             CHECK_EQ(static_cast<size_t>(pskv.size), size);
         }
         }
         return pskv;
     }

     void CheckUnique(const std::vector<int>& keys) {
        auto keys_copy = keys;
        auto last = std::unique(keys_copy.begin(), keys_copy.end());
        CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
                 static_cast<size_t>(keys.size()));
     }

     void InitNCCL(const std::vector<Context>& devs) {
          for (size_t i = 0; i < devs.size(); ++i) {
             device_ids_.push_back(devs[i].dev_id);
          }
          std::sort(device_ids_.begin(), device_ids_.end());
          std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
          std::vector<ncclComm_t> comms(devs.size());
          ncclCommInitAll(&(comms[0]), devs.size(), &(device_ids_[0]));
          for (size_t i = 0; i < devs.size(); ++i) {
             NCCLEntry e;
             e.dev_id = device_ids_[i];
             e.comm = comms[i];
             e.rank = i;
             cudaSetDevice(e.dev_id);
             cudaStreamCreate(&(e.stream));
             nccl_data_[device_ids_[i]] = e;
          }
          }

     //convert type to nccl type
     ncclDataType_t GetNCCLType(int dtype) {
        switch (dtype) {
        case mshadow::kFloat32:
            return ncclFloat;
        case mshadow::kFloat16:
            return ncclHalf;
        case mshadow::kFloat64:
            return ncclDouble;
        case mshadow::kUint8:
            return ncclChar;
        case mshadow::kInt32:
            return ncclInt;
        case mshadow::kInt64:
            return ncclInt64;
        default:
            LOG(FATAL) << "Unknown type passed to NCCL KVStore";
        }
        return ncclNumTypes;
     }
     //ps worker used for encode and send information
     ps::KVWorker<real_t>* ps_worker_;
     //sever_ pointer for new server start, worker node must not hold
     //a implemented one
     KVStoreDistServer* server_;

     //bufer for to copy from memory
     std::unordered_map<int, NDArray> merge_buf_;
     //bufer to merge between cards
     std::unordered_map<int, NDArray> merge_buf_devs_;
     //communication buffer
     std::unordered_map<int, NDArray> comm_buf_;
     //initialization flag
     bool nccl_init_;
     //nccl data
     struct NCCLEntry {
         /// \brief device ID
         int dev_id;
         /// \brief NCCL commmunicator
         ncclComm_t comm;
         /// \brief NCCL rank
         int rank;
         /// \brief GPU stream to use with NCCL
         cudaStream_t stream;
      };
     std::unordered_map<int, NCCLEntry> nccl_data_;

     //devs recording
     std::vector<Context> devs_;
     std::vector<int> device_ids_;

     //para size bound
     size_t bigarray_bound_;
};

} // kvstore
} // mxnet
#endif //NCCL
#endif //DIST NCCL